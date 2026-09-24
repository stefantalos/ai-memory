//! The LLM call ledger and breaker alarms (zero invisible spend).
//!
//! ref: floo scripts/lib/floo-metered-gate.mjs:62 (METERED_FUNDING_TYPE),
//!      :123-159 (authoriseMeteredCall ledgerRow shape) ·
//!      fn8-os scripts/mcp/lib/poolside-account.mjs:33 (key fingerprint)
//!
//! Every request a lane chain ([`crate::FallbackProvider`]) sends produces one
//! [`CallRecord`]: which lane (key) answered, for which caller, how many
//! tokens the provider reported, the outcome, and — for metered providers —
//! a cost estimate. PROJECTION: `cost_usd_est` is computed from the published
//! list price times the provider-reported token counts; it is an estimate,
//! not a billed figure. Every breaker that opens produces one
//! [`BreakerEvent`].
//!
//! [`JsonlLedger`] appends them as JSON lines:
//! * `calls_path` — every call and breaker event, the FinOps-readable ledger;
//! * `presence_path` — metered calls only, in the row shape of floo's
//!   `state/finops/harness-presence.jsonl` (the ledger `authoriseMeteredCall`
//!   writes), so ai-memory's Gemini spend lands where the charter counts it;
//! * `alarm_path` — breaker events in the chyros actionable-bus shape.
//!
//! Records never carry a key: a key is named only by the first 8 hex chars
//! of its sha256 ([`key_fingerprint`]). A ledger write failure is logged and
//! never fails a request.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::Serialize;
use sha2::{Digest, Sha256};

/// Funding label for pay-per-token rows, byte-identical to floo's
/// `METERED_FUNDING_TYPE` so its ledger readers classify them as metered.
pub const METERED_FUNDING_TYPE: &str = "Metered pay-per-token (billed per call)";

/// Published list price for a model, USD per 1M tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    /// Input (prompt) tokens.
    pub input_per_m: f64,
    /// Output (completion) tokens.
    pub output_per_m: f64,
}

/// Known list prices. `None` = not metered per token, or unknown.
///
/// gemini-2.5-flash, standard paid tier, text input — source:
/// <https://ai.google.dev/gemini-api/docs/pricing>, read 2026-09-24.
#[must_use]
pub fn published_price(provider: &str, model: &str) -> Option<Price> {
    let model = model.to_ascii_lowercase();
    if provider == "gemini" && model.starts_with("gemini-2.5-flash") && !model.contains("lite") {
        return Some(Price {
            input_per_m: 0.30,
            output_per_m: 2.50,
        });
    }
    None
}

/// Whether calls to this provider are billed per token.
#[must_use]
pub fn is_metered(provider: &str) -> bool {
    provider == "gemini"
}

/// Funding label for a provider/model, as recorded in the ledger.
#[must_use]
pub fn funding_for(provider: &str, model: &str) -> &'static str {
    if is_metered(provider) {
        METERED_FUNDING_TYPE
    } else if model.to_ascii_lowercase().starts_with("poolside/") {
        // CLAUDE.md §2.5: Poolside's funding model is UNVERIFIED.
        "Poolside developer preview (funding unverified)"
    } else {
        "unclassified"
    }
}

/// `sha256(key)[..8]` in hex — how a key is named in any record.
#[must_use]
pub fn key_fingerprint(key: &str) -> String {
    let digest = Sha256::digest(key.trim().as_bytes());
    digest.iter().take(4).map(|b| format!("{b:02x}")).collect()
}

/// One request sent to one lane.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CallRecord {
    /// RFC 3339 timestamp.
    pub ts: String,
    /// Record kind, always `"call"`.
    pub kind: &'static str,
    /// Operation that issued the request (`auto_improve`, `consolidate`, …).
    pub caller: String,
    /// Lane label (`key-a`, `key-b`, `gemini`, …).
    pub lane: String,
    /// Provider wire name.
    pub provider: String,
    /// Model id.
    pub model: String,
    /// Key fingerprint, when known.
    pub key_fp: Option<String>,
    /// `ok`, `quota`, `server`, `truncated`, `empty`, `fatal`.
    pub outcome: &'static str,
    /// HTTP status for a provider error.
    pub status: Option<u16>,
    /// Prompt tokens the provider reported.
    pub input_tokens: Option<u32>,
    /// Output tokens the provider reported.
    pub output_tokens: Option<u32>,
    /// HTTP requests this lane call sent (the token counts are their sum);
    /// `None` when the provider reported no usage.
    pub http_calls: Option<u32>,
    /// Estimate from [`published_price`] (see module docs); `None` when unpriced.
    pub cost_usd_est: Option<f64>,
    /// Funding label ([`funding_for`]).
    pub funding: &'static str,
}

/// A lane's breaker opened.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BreakerEvent {
    /// RFC 3339 timestamp.
    pub ts: String,
    /// Record kind, always `"breaker_open"`.
    pub kind: &'static str,
    /// Lane label.
    pub lane: String,
    /// Provider wire name.
    pub provider: String,
    /// Model id.
    pub model: String,
    /// `quota`, `server`, or `zero_yield`.
    pub reason: &'static str,
    /// Consecutive failures that tripped it (1 for a quota wall).
    pub consecutive: u32,
    /// Seconds the lane stays paused before a half-open probe.
    pub cooldown_secs: u64,
}

/// Receives call records and breaker events.
pub trait LaneObserver: Send + Sync {
    /// One request finished on one lane.
    fn on_call(&self, record: &CallRecord);
    /// A lane's breaker opened.
    fn on_breaker_open(&self, event: &BreakerEvent);
}

/// Estimate for one call from the list price (see module docs).
#[must_use]
pub fn estimate_cost(provider: &str, model: &str, input: u32, output: u32) -> Option<f64> {
    published_price(provider, model).map(|p| {
        (f64::from(input) * p.input_per_m + f64::from(output) * p.output_per_m) / 1_000_000.0
    })
}

/// Current time as RFC 3339.
#[must_use]
pub fn now_rfc3339() -> String {
    jiff::Timestamp::now().to_string()
}

/// Appends records as JSON lines. See module docs.
#[derive(Debug, Default)]
pub struct JsonlLedger {
    calls_path: Option<PathBuf>,
    presence_path: Option<PathBuf>,
    alarm_path: Option<PathBuf>,
    metered_entity: Option<String>,
    write_lock: Mutex<()>,
}

impl JsonlLedger {
    /// A ledger writing every record to `calls_path` (when set).
    #[must_use]
    pub fn new(calls_path: Option<PathBuf>) -> Self {
        Self {
            calls_path,
            ..Self::default()
        }
    }

    /// Also copy metered calls into a floo `harness-presence.jsonl`, booked
    /// to `entity` (`Floor No 8 SRL` or `Personal`).
    #[must_use]
    pub fn with_presence(mut self, path: Option<PathBuf>, entity: Option<String>) -> Self {
        self.presence_path = path;
        self.metered_entity = entity;
        self
    }

    /// Also append breaker events to an actionable-alarm bus.
    #[must_use]
    pub fn with_alarm(mut self, path: Option<PathBuf>) -> Self {
        self.alarm_path = path;
        self
    }

    fn append(&self, path: &PathBuf, line: &serde_json::Value) {
        let _guard = self
            .write_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut f| writeln!(f, "{line}"));
        if let Err(err) = result {
            tracing::warn!(path = %path.display(), error = %err, "LLM ledger write failed");
        }
    }
}

impl LaneObserver for JsonlLedger {
    fn on_call(&self, record: &CallRecord) {
        if let Some(path) = &self.calls_path {
            self.append(path, &serde_json::json!(record));
        }
        if let Some(path) = &self.presence_path
            && is_metered(&record.provider)
        {
            let entity = self
                .metered_entity
                .clone()
                .unwrap_or_else(|| "UNATTRIBUTED".into());
            let row = serde_json::json!({
                "timestamp": record.ts,
                "harness": record.provider,
                "harnessName": format!("{} (metered API) via ai-memory", record.provider),
                "model": record.model,
                "costCenter": entity,
                "fundingType": METERED_FUNDING_TYPE,
                "entity": entity,
                "taskId": format!("ai-memory:{}", record.caller),
                "operator": "Stefan",
                "notes": format!(
                    "ai-memory {} lane={} outcome={} http_calls={} in={} out={} est_usd={}",
                    record.caller,
                    record.lane,
                    record.outcome,
                    record.http_calls.map_or("?".into(), |t| t.to_string()),
                    record.input_tokens.map_or("?".into(), |t| t.to_string()),
                    record.output_tokens.map_or("?".into(), |t| t.to_string()),
                    record.cost_usd_est.map_or("?".into(), |c| format!("{c:.6}")),
                ),
            });
            self.append(path, &row);
        }
    }

    fn on_breaker_open(&self, event: &BreakerEvent) {
        if let Some(path) = &self.calls_path {
            self.append(path, &serde_json::json!(event));
        }
        if let Some(path) = &self.alarm_path {
            let row = serde_json::json!({
                "ts": event.ts,
                "actionable": true,
                "source": "ai-memory-llm",
                "severity": "MED",
                "reason": format!("llm-lane-paused:{}:{}", event.lane, event.reason),
                "recommended_action": format!(
                    "ai-memory paused LLM lane {} ({}/{}) for {}s after {} consecutive {} outcome(s); \
                     requests go to the next lane, and the lane is probed again after the cooldown.",
                    event.lane, event.provider, event.model, event.cooldown_secs,
                    event.consecutive, event.reason
                ),
            });
            self.append(path, &row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_eight_hex_of_sha256_and_ignores_surrounding_whitespace() {
        let fp = key_fingerprint("secret-value\n");
        assert_eq!(fp.len(), 8);
        assert_eq!(fp, key_fingerprint("secret-value"));
        assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn gemini_flash_estimate_uses_the_list_price() {
        // 35,441 in + 15,990 out: the measured truncated call.
        let c = estimate_cost("gemini", "gemini-2.5-flash", 35_441, 15_990).unwrap();
        assert!((c - 0.050_607_3).abs() < 1e-6, "{c}");
        assert_eq!(
            estimate_cost("openai-compat", "poolside/laguna-s-2.1", 1, 1),
            None
        );
    }

    #[test]
    fn metered_calls_are_copied_into_presence_rows_and_unmetered_are_not() {
        let dir = tempfile::tempdir().unwrap();
        let calls = dir.path().join("calls.jsonl");
        let presence = dir.path().join("presence.jsonl");
        let ledger = JsonlLedger::new(Some(calls.clone()))
            .with_presence(Some(presence.clone()), Some("Personal".into()));
        let mut rec = CallRecord {
            ts: now_rfc3339(),
            kind: "call",
            caller: "auto_improve".into(),
            lane: "gemini".into(),
            provider: "gemini".into(),
            model: "gemini-2.5-flash".into(),
            key_fp: Some("abcd1234".into()),
            outcome: "ok",
            status: None,
            input_tokens: Some(100),
            output_tokens: Some(10),
            http_calls: Some(1),
            cost_usd_est: estimate_cost("gemini", "gemini-2.5-flash", 100, 10),
            funding: funding_for("gemini", "gemini-2.5-flash"),
        };
        ledger.on_call(&rec);
        rec.provider = "openai-compat".into();
        rec.model = "poolside/laguna-s-2.1".into();
        ledger.on_call(&rec);
        let calls = std::fs::read_to_string(calls).unwrap();
        assert_eq!(calls.lines().count(), 2);
        let presence = std::fs::read_to_string(presence).unwrap();
        assert_eq!(presence.lines().count(), 1);
        let row: serde_json::Value =
            serde_json::from_str(presence.lines().next().unwrap()).unwrap();
        assert_eq!(row["harness"], "gemini", "only the metered call is copied");
        assert_eq!(row["model"], "gemini-2.5-flash");
        assert_eq!(row["fundingType"], METERED_FUNDING_TYPE);
        assert_eq!(row["entity"], "Personal");
        assert_eq!(row["taskId"], "ai-memory:auto_improve");
    }
}
