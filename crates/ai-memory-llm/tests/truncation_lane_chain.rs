//! Laguna truncation through the real lane chain, over real HTTP (wiremock).
//!
//! Measured 2026-09-24 in `~/.fn8/ai-memory/llm-calls.jsonl`: consolidate on
//! session d3345285 sent the same ~104k-token prompt to key-a ten times,
//! each call ending at the 14,000-token ceiling with no function call. Each
//! was recorded `fatal`, so the chain never moved to key-b or Gemini, and the
//! ledger showed only the LAST of the two HTTP calls every attempt made.
//!
//! These tests pin: a cut is a zero yield that moves to the next lane (never
//! a same-lane text retry); every HTTP call is booked; and an automated
//! metered fallback lands exactly one row in floo's `harness-presence.jsonl`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ai_memory_llm::types::ChatRequest;
use ai_memory_llm::{
    FallbackProvider, GateRequest, GateVerdict, GeminiProvider, JsonlLedger, Lane, LlmError,
    LlmProvider, MeteredGate, OpenAiCompatProvider, capture_responder, key_fingerprint,
    with_caller,
};
use async_trait::async_trait;
use secrecy::SecretString;
use serde_json::json;
use wiremock::matchers::{header, method};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const KEY_A: &str = "test-key-a-0123456789abcdef";
const KEY_B: &str = "test-key-b-0123456789abcdef";
const KEY_G: &str = "test-key-g-0123456789abcdef";

fn overthought(prompt: u32, completion: u32) -> serde_json::Value {
    json!({
        "choices": [{"message": {"role": "assistant", "content": null,
                                 "reasoning_content": "thinking..."},
                     "finish_reason": "length"}],
        "model": "poolside/laguna-s-2.1",
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion}
    })
}

fn prose(finish: &str, prompt: u32, completion: u32) -> serde_json::Value {
    json!({
        "choices": [{"message": {"role": "assistant", "content": "I will now list the pages"},
                     "finish_reason": finish}],
        "model": "poolside/laguna-s-2.1",
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion}
    })
}

/// Replays `responses` in order for one key, counting hits.
#[derive(Clone)]
struct Script {
    responses: Arc<Vec<serde_json::Value>>,
    hits: Arc<AtomicUsize>,
}

impl Respond for Script {
    fn respond(&self, _req: &Request) -> ResponseTemplate {
        let i = self.hits.fetch_add(1, Ordering::SeqCst);
        let body = &self.responses[i.min(self.responses.len() - 1)];
        ResponseTemplate::new(200).set_body_json(body.clone())
    }
}

async fn poolside_key(
    server: &MockServer,
    key: &str,
    responses: Vec<serde_json::Value>,
) -> Arc<AtomicUsize> {
    let hits = Arc::new(AtomicUsize::new(0));
    Mock::given(method("POST"))
        .and(header("authorization", format!("Bearer {key}").as_str()))
        .respond_with(Script {
            responses: Arc::new(responses),
            hits: hits.clone(),
        })
        .mount(server)
        .await;
    hits
}

async fn gemini_ok() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("x-goog-api-key", KEY_G))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": [{"text": "{\"summary\":\"from gemini\"}"}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 6646, "candidatesTokenCount": 900}
        })))
        .mount(&server)
        .await;
    server
}

/// A value gate that admits everything and counts how often it was asked.
#[derive(Default)]
struct AdmitGate(AtomicUsize);

#[async_trait]
impl MeteredGate for AdmitGate {
    async fn judge(&self, _request: GateRequest<'_>) -> GateVerdict {
        self.0.fetch_add(1, Ordering::SeqCst);
        GateVerdict {
            admit: true,
            reason: "test: admitted".into(),
            cost_usd: None,
        }
    }
}

fn poolside_lane(pool: &MockServer, key: &str) -> Arc<dyn LlmProvider> {
    Arc::new(
        OpenAiCompatProvider::new(
            pool.uri(),
            Some(SecretString::from(key.to_string())),
            "poolside/laguna-s-2.1",
        )
        .unwrap(),
    )
}

struct Paths {
    _dir: tempfile::TempDir,
    calls: std::path::PathBuf,
    presence: std::path::PathBuf,
}

fn paths() -> Paths {
    let dir = tempfile::tempdir().unwrap();
    Paths {
        calls: dir.path().join("llm-calls.jsonl"),
        presence: dir.path().join("harness-presence.jsonl"),
        _dir: dir,
    }
}

/// key-a, key-b, Gemini — the production lane order — with the ledger copying
/// metered calls into a presence file, and the value gate armed for the
/// production default callers.
fn production_chain(
    pool: &MockServer,
    gem: &MockServer,
    p: &Paths,
    gate: Arc<AdmitGate>,
) -> FallbackProvider {
    let g: Arc<dyn LlmProvider> = Arc::new(
        GeminiProvider::new(SecretString::from(KEY_G), "gemini-2.5-flash")
            .unwrap()
            .with_base_url(gem.uri()),
    );
    FallbackProvider::chain(
        Lane::new(poolside_lane(pool, KEY_A), "key-a").with_key_fp(Some(key_fingerprint(KEY_A))),
        vec![
            Lane::new(poolside_lane(pool, KEY_B), "key-b")
                .with_key_fp(Some(key_fingerprint(KEY_B))),
            Lane::new(g, "gemini").with_key_fp(Some(key_fingerprint(KEY_G))),
        ],
    )
    .with_observer(Arc::new(
        JsonlLedger::new(Some(p.calls.clone()))
            .with_presence(Some(p.presence.clone()), Some("Personal".into())),
    ))
    .with_metered_gate(gate, vec!["auto_improve".into(), "experience".into()])
}

fn schema() -> serde_json::Value {
    json!({"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]})
}

fn rows(path: &std::path::Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn call_rows(path: &std::path::Path) -> Vec<serde_json::Value> {
    rows(path)
        .into_iter()
        .filter(|r| r["kind"] == "call" && r["provider"] != "jev")
        .collect()
}

#[tokio::test]
async fn both_keys_cut_at_the_limit_reach_gemini_through_the_gate_and_book_one_presence_row() {
    let p = paths();
    let pool = MockServer::start().await;
    let hits_a = poolside_key(
        &pool,
        KEY_A,
        vec![overthought(6_646, 14_000), prose("stop", 1, 1)],
    )
    .await;
    let hits_b = poolside_key(
        &pool,
        KEY_B,
        vec![overthought(6_646, 14_000), prose("stop", 1, 1)],
    )
    .await;
    let gem = gemini_ok().await;
    let gate = Arc::new(AdmitGate::default());
    let chain = production_chain(&pool, &gem, &p, gate.clone());

    let (out, who) = capture_responder(with_caller(
        "auto_improve",
        chain.complete_structured_raw(ChatRequest::user_prompt("session"), schema()),
    ))
    .await;
    assert_eq!(out.expect("gemini answers")["summary"], "from gemini");
    assert_eq!(who.unwrap().lane, "gemini");

    // A cut moves the request on; it is never re-sent on the same lane.
    assert_eq!(
        hits_a.load(Ordering::SeqCst),
        1,
        "key-a: no same-lane text retry"
    );
    assert_eq!(
        hits_b.load(Ordering::SeqCst),
        1,
        "key-b: no same-lane text retry"
    );
    assert_eq!(
        gate.0.load(Ordering::SeqCst),
        1,
        "Jev judged the metered lane once"
    );

    let calls = call_rows(&p.calls);
    let outcomes: Vec<(&str, &str)> = calls
        .iter()
        .map(|r| (r["lane"].as_str().unwrap(), r["outcome"].as_str().unwrap()))
        .collect();
    assert_eq!(
        outcomes,
        vec![
            ("key-a", "truncated"),
            ("key-b", "truncated"),
            ("gemini", "ok")
        ]
    );
    assert_eq!(calls[0]["output_tokens"], 14_000);
    assert_eq!(calls[0]["http_calls"], 1);

    let presence = rows(&p.presence);
    assert_eq!(presence.len(), 1, "exactly one metered row: {presence:?}");
    let row = &presence[0];
    assert_eq!(row["entity"], "Personal");
    assert_eq!(row["costCenter"], "Personal");
    assert_eq!(row["model"], "gemini-2.5-flash");
    assert_eq!(row["harness"], "gemini");
    assert_eq!(row["taskId"], "ai-memory:auto_improve");
    assert_eq!(
        row["fundingType"],
        ai_memory_llm::ledger::METERED_FUNDING_TYPE
    );
    let notes = row["notes"].as_str().unwrap();
    assert!(
        notes.contains("lane=gemini") && notes.contains("in=6646"),
        "{notes}"
    );
    for key in [KEY_A, KEY_B, KEY_G] {
        let all = std::fs::read_to_string(&p.presence).unwrap()
            + &std::fs::read_to_string(&p.calls).unwrap();
        assert!(!all.contains(key), "a key value reached a ledger");
    }
}

/// Both HTTP calls of one lane call are booked with a correct total — not
/// just the last one. Distinct counts on each call so "keep last" and "keep
/// first" both fail. Poolside no longer sends the text fallback by default
/// (see the next test); the lane here opts back in to get two requests.
#[tokio::test]
async fn a_lane_call_that_sent_two_requests_is_booked_for_both() {
    let p = paths();
    let pool = MockServer::start().await;
    // Forced call stops with prose (a real shape miss) → text fallback, which
    // is then cut: two paid requests in one lane call.
    let hits = poolside_key(
        &pool,
        KEY_A,
        vec![prose("stop", 100, 7_000), prose("length", 120, 14_000)],
    )
    .await;
    let lane: Arc<dyn LlmProvider> = Arc::new(
        OpenAiCompatProvider::new(
            pool.uri(),
            Some(SecretString::from(KEY_A.to_string())),
            "poolside/laguna-s-2.1",
        )
        .unwrap()
        .with_tool_text_fallback(true),
    );
    let chain = FallbackProvider::chain(Lane::new(lane, "key-a"), Vec::new())
        .with_observer(Arc::new(JsonlLedger::new(Some(p.calls.clone()))));

    let err = with_caller(
        "consolidate",
        chain.complete_structured_raw(ChatRequest::user_prompt("session"), schema()),
    )
    .await
    .expect_err("cut");
    assert!(matches!(err, LlmError::Truncated { .. }), "{err:?}");
    assert_eq!(hits.load(Ordering::SeqCst), 2);

    let calls = call_rows(&p.calls);
    assert_eq!(calls.len(), 1, "one lane call, one row");
    assert_eq!(calls[0]["http_calls"], 2);
    assert_eq!(calls[0]["input_tokens"], 220);
    assert_eq!(calls[0]["output_tokens"], 21_000);
    assert_eq!(calls[0]["outcome"], "truncated");
}

/// Measured 2026-09-24 on b50cd18b: consolidate booked http_calls=2 and
/// ~16.6k output three times (a prose reply, then the thinking-on text
/// fallback cut at 14,000). By default a Poolside lane now sends ONE request:
/// the prose reply is a shape error, booked once, and no fallback is paid for.
#[tokio::test]
async fn a_poolside_prose_reply_is_one_request_and_one_booking() {
    let p = paths();
    let pool = MockServer::start().await;
    let hits = poolside_key(
        &pool,
        KEY_A,
        vec![prose("stop", 100, 7_000), prose("length", 120, 14_000)],
    )
    .await;
    let chain =
        FallbackProvider::chain(Lane::new(poolside_lane(&pool, KEY_A), "key-a"), Vec::new())
            .with_observer(Arc::new(JsonlLedger::new(Some(p.calls.clone()))));
    let err = with_caller(
        "consolidate",
        chain.complete_structured_raw(ChatRequest::user_prompt("session"), schema()),
    )
    .await
    .expect_err("prose is not a result");
    assert!(matches!(err, LlmError::UnexpectedShape(_)), "{err:?}");
    assert_eq!(hits.load(Ordering::SeqCst), 1, "no text fallback request");
    let calls = call_rows(&p.calls);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["http_calls"], 1);
    assert_eq!(calls[0]["output_tokens"], 7_000);
}

/// Pins a policy gap rather than blessing it: `consolidate` is NOT in the
/// default gated callers, so once a cut routes past both Poolside keys it
/// reaches metered Gemini without asking Jev. The spend is still booked in
/// presence. Adding `consolidate` to `AI_MEMORY_METERED_GATE_CALLERS` is the
/// operator's switch; when that default changes, this test must change too.
#[tokio::test]
async fn consolidate_is_not_gated_by_default_but_its_metered_spend_is_still_booked() {
    let p = paths();
    let pool = MockServer::start().await;
    poolside_key(&pool, KEY_A, vec![overthought(104_054, 14_000)]).await;
    poolside_key(&pool, KEY_B, vec![overthought(104_054, 14_000)]).await;
    let gem = gemini_ok().await;
    let gate = Arc::new(AdmitGate::default());
    let chain = production_chain(&pool, &gem, &p, gate.clone());

    let out = with_caller(
        "consolidate",
        chain.complete_structured_raw(ChatRequest::user_prompt("session"), schema()),
    )
    .await;
    assert_eq!(out.unwrap()["summary"], "from gemini");
    assert_eq!(
        gate.0.load(Ordering::SeqCst),
        0,
        "consolidate is not a gated caller"
    );
    let presence = rows(&p.presence);
    assert_eq!(presence.len(), 1);
    assert_eq!(presence[0]["taskId"], "ai-memory:consolidate");
}
