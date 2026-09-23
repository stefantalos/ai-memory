//! A Gemini structured response cut at maxOutputTokens is `Truncated`, not a
//! serde error (measured 2026-09-23: a 294-observation auto_improve run died
//! as a bare `serde: EOF while parsing a string` 502, no run recorded).

use ai_memory_llm::types::ChatRequest;
use ai_memory_llm::{GeminiProvider, LlmError, LlmProvider};
use secrecy::SecretString;
use serde_json::json;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn gemini_returning(text: &str, finish: &str) -> GeminiProvider {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{ "content": { "parts": [{ "text": text }] }, "finishReason": finish }]
        })))
        .mount(&server)
        .await;
    let base = server.uri();
    std::mem::forget(server);
    GeminiProvider::new(SecretString::from("k"), "gemini-2.5-flash")
        .unwrap()
        .with_base_url(base)
}

fn schema() -> serde_json::Value {
    json!({ "type": "object", "properties": { "summary": { "type": "string" } }, "required": ["summary"] })
}

#[tokio::test]
async fn max_tokens_cut_is_truncated() {
    let p = gemini_returning("{\n  \"summary\": \"a very long unfinished", "MAX_TOKENS").await;
    let err = p
        .complete_structured_raw(ChatRequest::user_prompt("x"), schema())
        .await
        .unwrap_err();
    assert!(matches!(err, LlmError::Truncated { .. }), "got {err:?}");
}

#[tokio::test]
async fn bad_json_that_stopped_normally_stays_a_parse_error() {
    let p = gemini_returning("{ not json", "STOP").await;
    let err = p
        .complete_structured_raw(ChatRequest::user_prompt("x"), schema())
        .await
        .unwrap_err();
    assert!(!matches!(err, LlmError::Truncated { .. }), "got {err:?}");
}

#[tokio::test]
async fn complete_json_is_returned() {
    let p = gemini_returning("{\"summary\": \"ok\"}", "STOP").await;
    let v = p
        .complete_structured_raw(ChatRequest::user_prompt("x"), schema())
        .await
        .unwrap();
    assert_eq!(v["summary"], json!("ok"));
}
