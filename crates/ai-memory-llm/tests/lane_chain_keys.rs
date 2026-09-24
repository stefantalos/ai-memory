//! Two Poolside keys ahead of Gemini, over real HTTP (wiremock): a limited key
//! moves the SAME request to the next key, both limited reaches Gemini, and no
//! key value ever appears in a log line, an error, or the call ledger.

use std::io::Write;
use std::sync::{Arc, Mutex};

use ai_memory_llm::types::ChatRequest;
use ai_memory_llm::{
    FallbackProvider, GeminiProvider, JsonlLedger, Lane, LlmError, LlmProvider,
    OpenAiCompatProvider, capture_responder, key_fingerprint, with_caller,
};
use secrecy::SecretString;
use serde_json::json;
use wiremock::matchers::{header, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

const KEY_A: &str = "sentinel-key-AAAA-0123456789abcdef";
const KEY_B: &str = "sentinel-key-BBBB-0123456789abcdef";
const KEY_G: &str = "sentinel-key-GGGG-0123456789abcdef";

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);
impl Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Captured {
    type Writer = Captured;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// One mock host answering per key: `limited` keys get Poolside's real 429.
async fn poolside(limited: &[&str], ok_key: Option<&str>) -> MockServer {
    let server = MockServer::start().await;
    for key in limited {
        Mock::given(method("POST"))
            .and(header("authorization", format!("Bearer {key}").as_str()))
            .respond_with(
                ResponseTemplate::new(429).set_body_json(json!({"error": "usage limit exceeded"})),
            )
            .mount(&server)
            .await;
    }
    if let Some(key) = ok_key {
        Mock::given(method("POST"))
            .and(header("authorization", format!("Bearer {key}").as_str()))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "", "arguments": "{\"summary\":\"from key b\"}"}
                }]}, "finish_reason": "tool_calls"}],
                "model": "poolside/laguna-s-2.1",
                "usage": {"prompt_tokens": 12, "completion_tokens": 3}
            })))
            .mount(&server)
            .await;
    }
    server
}

async fn gemini(status: u16) -> MockServer {
    let server = MockServer::start().await;
    let body = if status == 200 {
        json!({
            "candidates": [{"content": {"parts": [{"text": "{\"summary\":\"from gemini\"}"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 5}
        })
    } else {
        json!({"error": {"code": status, "message": "quota"}})
    };
    Mock::given(method("POST"))
        .and(header("x-goog-api-key", KEY_G))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(&server)
        .await;
    server
}

fn chain(pool: &MockServer, gem: &MockServer, ledger: &std::path::Path) -> FallbackProvider {
    let lane = |key: &str| -> Arc<dyn LlmProvider> {
        Arc::new(
            OpenAiCompatProvider::new(
                pool.uri(),
                Some(SecretString::from(key.to_string())),
                "poolside/laguna-s-2.1",
            )
            .unwrap(),
        )
    };
    let g: Arc<dyn LlmProvider> = Arc::new(
        GeminiProvider::new(SecretString::from(KEY_G), "gemini-2.5-flash")
            .unwrap()
            .with_base_url(gem.uri()),
    );
    FallbackProvider::chain(
        Lane::new(lane(KEY_A), "key-a").with_key_fp(Some(key_fingerprint(KEY_A))),
        vec![
            Lane::new(lane(KEY_B), "key-b").with_key_fp(Some(key_fingerprint(KEY_B))),
            Lane::new(g, "gemini").with_key_fp(Some(key_fingerprint(KEY_G))),
        ],
    )
    .with_observer(Arc::new(JsonlLedger::new(Some(ledger.to_path_buf()))))
}

fn schema() -> serde_json::Value {
    json!({"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]})
}

fn assert_no_key(haystack: &str, what: &str) {
    for key in [KEY_A, KEY_B, KEY_G] {
        assert!(!haystack.contains(key), "{what} leaked a key value");
        // Not even a recognisable fragment of one.
        assert!(!haystack.contains(&key[13..29]), "{what} leaked a key fragment");
    }
}

#[tokio::test]
async fn limited_key_a_moves_to_key_b_and_no_key_is_logged() {
    let logs = Captured::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(logs.clone())
        .with_max_level(tracing::Level::TRACE)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let dir = tempfile::tempdir().unwrap();
    let ledger = dir.path().join("calls.jsonl");
    let pool = poolside(&[KEY_A], Some(KEY_B)).await;
    let gem = gemini(200).await;
    let f = chain(&pool, &gem, &ledger);

    let (out, who) = capture_responder(with_caller(
        "auto_improve",
        f.complete_structured_raw(ChatRequest::user_prompt("x"), schema()),
    ))
    .await;
    assert_eq!(out.unwrap()["summary"], "from key b");
    assert_eq!(who.unwrap().lane, "key-b");

    let log_text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    assert!(log_text.contains("key-a"), "the lane switch is logged: {log_text}");
    assert_no_key(&log_text, "logs");
    let ledger_text = std::fs::read_to_string(&ledger).unwrap();
    assert!(ledger_text.contains(&key_fingerprint(KEY_A)));
    assert!(ledger_text.contains("\"outcome\":\"quota\""));
    assert_no_key(&ledger_text, "ledger");
}

#[tokio::test]
async fn both_keys_limited_reach_gemini_and_errors_carry_no_key() {
    let logs = Captured::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(logs.clone())
        .with_max_level(tracing::Level::TRACE)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let dir = tempfile::tempdir().unwrap();
    let ledger = dir.path().join("calls.jsonl");
    let pool = poolside(&[KEY_A, KEY_B], None).await;

    let gem_ok = gemini(200).await;
    let f = chain(&pool, &gem_ok, &ledger);
    let (out, who) = capture_responder(
        f.complete_structured_raw(ChatRequest::user_prompt("x"), schema()),
    )
    .await;
    assert_eq!(out.unwrap()["summary"], "from gemini");
    assert_eq!(who.unwrap().lane, "gemini");

    // All three limited: the error the caller sees names no key.
    let gem_limited = gemini(429).await;
    let f = chain(&pool, &gem_limited, &ledger);
    let err = f
        .complete_structured_raw(ChatRequest::user_prompt("x"), schema())
        .await
        .unwrap_err();
    assert!(matches!(err, LlmError::Provider { status: 429, .. }), "{err:?}");
    assert_no_key(&format!("{err} {err:?}"), "error");
    // And once every breaker is open, nothing is sent at all.
    let paused = f
        .complete_structured_raw(ChatRequest::user_prompt("x"), schema())
        .await
        .unwrap_err();
    assert!(matches!(paused, LlmError::LanesPaused(_)), "{paused:?}");
    assert_no_key(&format!("{paused} {paused:?}"), "paused error");

    let log_text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    assert!(log_text.contains("ALARM"), "breaker alarms are logged");
    assert_no_key(&log_text, "logs");
    let ledger_text = std::fs::read_to_string(&ledger).unwrap();
    assert!(ledger_text.contains("\"kind\":\"breaker_open\""));
    assert!(ledger_text.contains("\"input_tokens\":40"), "gemini usage recorded");
    assert_no_key(&ledger_text, "ledger");
}

/// The measured Gemini fix on the wire: the request pins propertyOrdering.
#[tokio::test]
async fn gemini_request_pins_property_order() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(wiremock::matchers::body_partial_json(json!({
            "generationConfig": {"responseSchema": {"propertyOrdering": ["proposals", "summary"]}}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": [{"text": "{\"proposals\":[],\"summary\":\"s\"}"}]}, "finishReason": "STOP"}]
        })))
        .mount(&server)
        .await;
    let g = GeminiProvider::new(SecretString::from("k"), "gemini-2.5-flash")
        .unwrap()
        .with_base_url(server.uri());
    let schema = json!({"type": "object", "properties": {
        "proposals": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"}
    }});
    let v = g
        .complete_structured_raw(ChatRequest::user_prompt("x"), schema)
        .await
        .expect("mock matched only if propertyOrdering was sent");
    assert_eq!(v["summary"], "s");
}
