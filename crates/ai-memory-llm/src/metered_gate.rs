//! Value gate in front of metered lanes: Jev decides whether a request is
//! worth paying for before it reaches a pay-per-token provider.
//!
//! ref: floo scripts/bin/floo-jev.mjs:170-175 (`--noul` custom primitive),
//!      :216-217 (`--json` output) · scripts/lib/floo-typesafe-jev.mjs:503-525
//!      (emulator fallback when no key; `authoriseMeteredCall` on live calls)
//!
//! Operator decision 2026-09-24 ("put Jev on it to route it"). The split:
//! * availability routing — key A, key B, Gemini, cooldowns, breakers — is
//!   deterministic code in [`crate::FallbackProvider`], never a Jev question;
//! * VALUE is Jev's: before a gated caller's request reaches a metered lane,
//!   Jev answers atomic yes/no questions over the request's evidence, and the
//!   answers are composed here in code. Every answer must clear the money
//!   threshold ([`MONEY_THRESHOLD`], 0.9). A "no", an "unknown", an emulated
//!   answer, a crash or a timeout all decline: the metered call is not made
//!   (fail closed) and the request waits for a free lane.
//!
//! [`JevCliGate`] reuses the existing floo Jev CLI (`floo-jev.mjs --noul …
//! --live --json`) — one process per question — rather than building a new
//! judge. The CLI records its own OpenRouter spend through floo's
//! `authoriseMeteredCall`; the gate also reports the cost it read back so
//! the ai-memory ledger carries it.

use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;

/// Jev's money bar: every atomic judgement must reach it.
pub const MONEY_THRESHOLD: f64 = 0.9;

/// What the gate is asked about.
#[derive(Debug, Clone, Copy)]
pub struct GateRequest<'a> {
    /// Operation that issued the request (`auto_improve`, …).
    pub caller: &'a str,
    /// Metered provider about to be called.
    pub provider: &'a str,
    /// Its model.
    pub model: &'a str,
    /// The request's evidence (the user prompt), already bounded.
    pub evidence: &'a str,
}

/// The gate's decision.
#[derive(Debug, Clone, PartialEq)]
pub struct GateVerdict {
    /// Whether the metered call may be made.
    pub admit: bool,
    /// Human-readable reason (scores, or why the judge was unavailable).
    pub reason: String,
    /// What asking cost, when the judge reported it (USD).
    pub cost_usd: Option<f64>,
}

impl GateVerdict {
    /// A decline.
    #[must_use]
    pub fn decline(reason: impl Into<String>) -> Self {
        Self {
            admit: false,
            reason: reason.into(),
            cost_usd: None,
        }
    }
}

/// Decides whether a metered call is worth making.
#[async_trait]
pub trait MeteredGate: Send + Sync {
    /// Judge one request. Must never panic; any failure is a decline.
    async fn judge(&self, request: GateRequest<'_>) -> GateVerdict;
}

/// The atomic questions, one judgement each, composed by [`JevCliGate`].
/// Positive phrasing so a high score always means "worth paying for". The
/// set follows the operator's correction of 2026-09-24: never "is it an
/// implementation detail" — a gotcha tied to a specific file or env var is
/// valuable precisely because the code does not reveal it — but (a) will it
/// be needed again, (b) is it NOT cheaply rediscoverable from code or git,
/// (c) is it not already documented.
pub const VALUE_QUESTIONS: &[(&str, &str)] = &[
    (
        "future_need",
        "Would a future agent working in this project hit the problem described in this session \
         record again, or need one of the facts it establishes again?",
    ),
    (
        "not_rediscoverable",
        "Would that problem or fact stay hidden from someone who reads the code or the git \
         history — for example a trap caused by an unset environment variable, a runtime or \
         vendor behaviour, or a validation rule the code does not reveal — so that rediscovering \
         it would be costly?",
    ),
    (
        "not_already_documented",
        "Is that problem or fact absent from the existing wiki pages this record lists, so that \
         writing it down would add knowledge rather than repeat it?",
    ),
];

/// Characters of evidence handed to Jev.
pub const MAX_GATE_EVIDENCE_CHARS: usize = 24_000;

/// Head-bounded evidence, cut on a char boundary.
#[must_use]
pub fn bound_evidence(text: &str) -> &str {
    if text.len() <= MAX_GATE_EVIDENCE_CHARS {
        return text;
    }
    let mut end = MAX_GATE_EVIDENCE_CHARS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Jev via the floo CLI. See module docs.
#[derive(Debug, Clone)]
pub struct JevCliGate {
    node: PathBuf,
    script: PathBuf,
    timeout: Duration,
    threshold: f64,
}

impl JevCliGate {
    /// `node` runs `script` (`floo-jev.mjs`).
    #[must_use]
    pub fn new(node: PathBuf, script: PathBuf) -> Self {
        Self {
            node,
            script,
            timeout: Duration::from_secs(60),
            threshold: MONEY_THRESHOLD,
        }
    }

    /// Per-question timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    async fn ask(&self, question: &str, evidence: &str) -> Result<(f64, Option<f64>), String> {
        let child = tokio::process::Command::new(&self.node)
            .arg(&self.script)
            .arg("--noul")
            .arg(question)
            .arg("--state")
            .arg(evidence)
            .arg("--live")
            .arg("--json")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .output();
        let output = tokio::time::timeout(self.timeout, child)
            .await
            .map_err(|_| format!("timed out after {}s", self.timeout.as_secs()))?
            .map_err(|e| format!("could not run: {e}"))?;
        if !output.status.success() {
            return Err(format!("exited with {}", output.status));
        }
        let body: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| format!("unreadable output: {e}"))?;
        let cost = body
            .get("usage")
            .and_then(|u| u.get("cost"))
            .and_then(serde_json::Value::as_f64);
        let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("");
        if model.contains("emulator") {
            // No key, or the FinOps gate refused: the CLI answered from its
            // local emulator. That is no judgement at all.
            return Err("answered by the local emulator, not the model".into());
        }
        let noul = body
            .pointer("/answers/custom_noul/noul")
            .and_then(serde_json::Value::as_f64)
            .filter(|n| n.is_finite())
            .ok_or_else(|| "no noul answer".to_string())?;
        Ok((noul, cost))
    }
}

#[async_trait]
impl MeteredGate for JevCliGate {
    async fn judge(&self, request: GateRequest<'_>) -> GateVerdict {
        let evidence = bound_evidence(request.evidence);
        let mut cost = 0.0;
        let mut any_cost = false;
        let mut scores = Vec::new();
        for (id, question) in VALUE_QUESTIONS {
            match self.ask(question, evidence).await {
                Ok((noul, c)) => {
                    if let Some(c) = c {
                        cost += c;
                        any_cost = true;
                    }
                    scores.push(format!("{id}={noul:.2}"));
                    if noul < self.threshold {
                        return GateVerdict {
                            admit: false,
                            reason: format!(
                                "jev declined: {id}={noul:.2} < {} ({})",
                                self.threshold,
                                scores.join(", ")
                            ),
                            cost_usd: any_cost.then_some(cost),
                        };
                    }
                }
                Err(why) => {
                    return GateVerdict {
                        admit: false,
                        reason: format!("jev unavailable on {id}: {why} (fail closed)"),
                        cost_usd: any_cost.then_some(cost),
                    };
                }
            }
        }
        GateVerdict {
            admit: true,
            reason: format!("jev admitted: {}", scores.join(", ")),
            cost_usd: any_cost.then_some(cost),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// A fake `node` that ignores the script and prints a scripted answer
    /// per question (matched on a word in the question text).
    fn fake_node(dir: &std::path::Path, body: &str) -> PathBuf {
        let path = dir.join("node");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn answer(noul: f64, model: &str) -> String {
        format!(
            "printf '%s' '{{\"model\":\"{model}\",\"answers\":{{\"custom_noul\":{{\"type\":\"noul\",\"noul\":{noul}}}}},\"usage\":{{\"cost\":0.0001}}}}'"
        )
    }

    fn req() -> GateRequest<'static> {
        GateRequest {
            caller: "auto_improve",
            provider: "gemini",
            model: "gemini-2.5-flash",
            evidence: "session evidence",
        }
    }

    async fn judge_with(body: &str) -> GateVerdict {
        let dir = tempfile::tempdir().unwrap();
        let node = fake_node(dir.path(), body);
        JevCliGate::new(node, "unused.mjs".into())
            .with_timeout(Duration::from_secs(5))
            .judge(req())
            .await
    }

    #[tokio::test]
    async fn every_answer_above_the_money_bar_admits_and_reports_cost() {
        let v = judge_with(&answer(0.95, "typesafe/jev-1.13")).await;
        assert!(v.admit, "{v:?}");
        assert!((v.cost_usd.unwrap() - 0.0003).abs() < 1e-9, "three questions asked");
    }

    #[tokio::test]
    async fn one_answer_below_the_money_bar_declines() {
        let v = judge_with(&answer(0.89, "typesafe/jev-1.13")).await;
        assert!(!v.admit);
        assert!(v.reason.contains("future_need=0.89"), "{}", v.reason);
    }

    #[tokio::test]
    async fn a_later_question_can_decline_alone() {
        let body = format!(
            "case \"$3\" in *hidden*) {} ;; *) {} ;; esac",
            answer(0.5, "typesafe/jev-1.13"),
            answer(0.97, "typesafe/jev-1.13")
        );
        let v = judge_with(&body).await;
        assert!(!v.admit);
        assert!(v.reason.contains("not_rediscoverable=0.50"), "{}", v.reason);
    }

    #[tokio::test]
    async fn an_emulated_answer_is_no_judgement() {
        let v = judge_with(&answer(0.99, "jev-emulator-v1")).await;
        assert!(!v.admit);
        assert!(v.reason.contains("emulator"), "{}", v.reason);
    }

    #[tokio::test]
    async fn crash_garbage_and_timeout_fail_closed() {
        assert!(!judge_with("exit 3").await.admit);
        assert!(!judge_with("echo not-json").await.admit);
        let dir = tempfile::tempdir().unwrap();
        let node = fake_node(dir.path(), "sleep 5");
        let v = JevCliGate::new(node, "x".into())
            .with_timeout(Duration::from_millis(200))
            .judge(req())
            .await;
        assert!(!v.admit);
        assert!(v.reason.contains("timed out"), "{}", v.reason);
        let missing = JevCliGate::new("/nonexistent/node".into(), "x".into())
            .judge(req())
            .await;
        assert!(!missing.admit);
    }

    #[test]
    fn evidence_is_head_bounded_on_a_char_boundary() {
        let long = "ș".repeat(MAX_GATE_EVIDENCE_CHARS);
        let b = bound_evidence(&long);
        assert!(b.len() <= MAX_GATE_EVIDENCE_CHARS);
        assert!(long.starts_with(b));
    }
}
