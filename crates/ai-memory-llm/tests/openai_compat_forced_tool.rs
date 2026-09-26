//! Forced-tool structured output for Poolside/Laguna, against a wiremock
//! upstream.
//!
//! Measured live 2026-09-23 on inference.poolside.ai with
//! `poolside/laguna-s-2.1`: `response_format` with a strict json_schema was
//! ignored 4 of 4 times, while one forced function call with
//! `chat_template_kwargs.enable_thinking=false` returned a schema-conformant
//! object 9 of 11 times; the other two carried a nested array as a
//! JSON-encoded string. These tests pin the request shape and the parsing of
//! the Laguna contract (FN8-9336, `crates/ai-memory-llm/src/laguna.rs`) when
//! the host answers with one JSON body instead of a stream; the streamed path
//! is in `zero_metered_chain.rs`.

use ai_memory_llm::types::ChatRequest;
use ai_memory_llm::{LlmProvider, OpenAiCompatProvider};
use serde_json::json;
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

fn schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "summary": { "type": "string" },
            "proposals": {
                "type": "array",
                "items": { "$ref": "#/$defs/Proposal" }
            }
        },
        "required": ["summary", "proposals"],
        "$defs": {
            "Proposal": {
                "type": "object",
                "properties": {
                    "rationale": { "type": "string" },
                    "evidence": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["rationale", "evidence"]
            }
        }
    })
}

fn tool_call_body(arguments: serde_json::Value) -> serde_json::Value {
    json!({
        "id": "id", "object": "chat.completion", "created": 0, "model": "poolside/laguna-s-2.1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant", "content": null,
                "tool_calls": [{ "id": "c1", "type": "function",
                    "function": { "name": "submit_structured_output", "arguments": arguments } }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn text_body(content: &str) -> serde_json::Value {
    json!({
        "id": "id", "object": "chat.completion", "created": 0, "model": "poolside/laguna-s-2.1",
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": content }, "finish_reason": "stop" }]
    })
}

/// Replays `responses` in order and records every request body.
#[derive(Clone)]
struct Script {
    responses: Arc<Vec<serde_json::Value>>,
    seen: Arc<Mutex<Vec<serde_json::Value>>>,
}

impl Respond for Script {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let mut seen = self.seen.lock().unwrap();
        seen.push(serde_json::from_slice(&req.body).unwrap());
        let i = (seen.len() - 1).min(self.responses.len() - 1);
        ResponseTemplate::new(200).set_body_json(self.responses[i].clone())
    }
}

async fn serve(
    responses: Vec<serde_json::Value>,
) -> (MockServer, Arc<Mutex<Vec<serde_json::Value>>>) {
    let server = MockServer::start().await;
    let seen = Arc::new(Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Script {
            responses: Arc::new(responses),
            seen: seen.clone(),
        })
        .mount(&server)
        .await;
    (server, seen)
}

fn provider(base: &str, model: &str) -> OpenAiCompatProvider {
    OpenAiCompatProvider::new(base.to_string(), None, model)
        .expect("provider builds")
        .with_strict(true)
}

fn request() -> ChatRequest {
    ChatRequest::user_prompt("review these observations")
}

#[tokio::test]
async fn poolside_model_forces_one_tool_call_without_response_format() {
    let args = json!({ "summary": "s", "proposals": [{ "rationale": "r", "evidence": ["e"] }] });
    let (server, seen) = serve(vec![tool_call_body(json!(args.to_string()))]).await;
    let out = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .complete_structured_raw(request(), schema())
        .await
        .expect("structured result");
    assert_eq!(out, args);

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "one call, no fallback");
    let body = &seen[0];
    assert_eq!(body["stream"], json!(true), "the contract streams the forced call");
    assert_eq!(body["temperature"], json!(0.7));
    assert!(
        body.get("response_format").is_none(),
        "response_format is ignored upstream; not sent"
    );
    assert_eq!(
        body["tool_choice"]["function"]["name"],
        json!("submit_structured_output")
    );
    assert_eq!(
        body["tools"][0]["function"]["name"],
        json!("submit_structured_output")
    );
    assert_eq!(body["tools"][0]["function"]["parameters"], schema());
    assert!(
        body.get("chat_template_kwargs").is_none(),
        "a non-Poolside host gets no thinking toggle"
    );
}

#[tokio::test]
async fn thinking_toggle_is_sent_on_the_forced_call_only() {
    let args = json!({ "summary": "s", "proposals": [] });
    let (server, seen) = serve(vec![tool_call_body(args.clone()), text_body("plain")]).await;
    let p = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .with_disable_thinking(true);
    p.complete_structured_raw(request(), schema())
        .await
        .expect("structured");
    p.complete(request()).await.expect("text");
    let seen = seen.lock().unwrap();
    assert_eq!(
        seen[0]["chat_template_kwargs"],
        json!({ "enable_thinking": false })
    );
    assert!(
        seen[1].get("chat_template_kwargs").is_none(),
        "free-text calls keep the model default"
    );
}

#[tokio::test]
async fn a_nested_array_sent_as_a_json_string_is_decoded() {
    let stringified = json!({
        "summary": "s",
        "proposals": json!([{ "rationale": "r", "evidence": ["e"] }]).to_string()
    });
    let (server, _) = serve(vec![tool_call_body(json!(stringified.to_string()))]).await;
    let out = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .complete_structured_raw(request(), schema())
        .await
        .expect("structured result");
    assert!(
        out["proposals"].is_array(),
        "stringified array decoded: {out}"
    );
    assert_eq!(out["proposals"][0]["evidence"][0], json!("e"));
}

#[tokio::test]
async fn a_string_field_that_looks_like_json_is_left_alone() {
    let args = json!({ "summary": "[1, 2]", "proposals": [] });
    let (server, _) = serve(vec![tool_call_body(args.clone())]).await;
    let out = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .complete_structured_raw(request(), schema())
        .await
        .expect("structured result");
    assert_eq!(
        out["summary"],
        json!("[1, 2]"),
        "schema says string: never decoded"
    );
}

/// No call, no JSON: one more sample of the SAME forced request at 0.9 —
/// never a plain text call with thinking at the template default (cut 4 of 4
/// in the A/B). The second sample's answer text holds a valid object, which
/// the contract salvages.
#[tokio::test]
async fn no_tool_call_retries_the_forced_call_once_and_salvages_valid_json() {
    let (server, seen) = serve(vec![
        text_body("I'll propose two pages."),
        text_body("here: {\"summary\": \"s\", \"proposals\": []}"),
    ])
    .await;
    let out = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .with_disable_thinking(true)
        .complete_structured_raw(request(), schema())
        .await
        .expect("salvaged result");
    assert_eq!(out, json!({ "summary": "s", "proposals": [] }));
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    for body in seen.iter() {
        assert_eq!(body["tool_choice"]["function"]["name"], json!("submit_structured_output"));
        assert_eq!(body["chat_template_kwargs"], json!({ "enable_thinking": false }));
    }
    assert_eq!((seen[0]["temperature"].clone(), seen[1]["temperature"].clone()), (json!(0.7), json!(0.9)));
}

/// Salvage is strict: an object in the text that fails the schema is not an
/// answer.
#[tokio::test]
async fn an_invalid_object_in_the_text_is_not_salvaged() {
    let (server, seen) = serve(vec![
        text_body("{\"summary\": 3}"),
        text_body("{\"proposals\": []}"),
    ])
    .await;
    let err = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .complete_structured_raw(request(), schema())
        .await
        .expect_err("nothing valid");
    assert!(matches!(err, ai_memory_llm::LlmError::UnexpectedShape(_)), "{err:?}");
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn other_models_keep_response_format() {
    let (server, seen) = serve(vec![text_body("{\"summary\": \"s\", \"proposals\": []}")]).await;
    let _ = provider(&format!("{}/v1", server.uri()), "mistral-nemo")
        .complete_structured_raw(request(), schema())
        .await
        .expect("structured result");
    let seen = seen.lock().unwrap();
    assert!(seen[0].get("response_format").is_some());
    assert!(seen[0].get("tools").is_none());
}

#[tokio::test]
async fn poolside_structured_output_is_capped_at_4k_and_text_at_14k() {
    let (server, seen) = serve(vec![
        tool_call_body(json!({ "summary": "s", "proposals": [] })),
        text_body("plain"),
        text_body("{\"summary\": \"s\", \"proposals\": []}"),
    ])
    .await;
    let base = format!("{}/v1", server.uri());
    let mut big = request();
    big.max_tokens = 32_000;
    let p = provider(&base, "poolside/laguna-s-2.1");
    p.complete_structured_raw(big.clone(), schema())
        .await
        .unwrap();
    p.complete(big.clone()).await.unwrap();
    provider(&base, "mistral-nemo")
        .complete_structured_raw(big.clone(), schema())
        .await
        .unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0]["max_tokens"], json!(4_000), "structured call: the Laguna contract ceiling");
    assert_eq!(seen[1]["max_tokens"], json!(14_000), "text call capped");
    assert_eq!(
        seen[2]["max_tokens"],
        json!(32_000),
        "other models untouched"
    );
}

/// Laguna's documented overthinking: the whole output budget goes to
/// reasoning, no function call, `finish_reason: "length"`.
fn overthought_body(prompt: u32, completion: u32) -> serde_json::Value {
    json!({
        "id": "id", "object": "chat.completion", "created": 0, "model": "poolside/laguna-s-2.1",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": null,
                         "reasoning_content": "Draft: {\"summary\": \"s\", \"proposals\": [{\"rationale\": \"r\", \"evidence\": [\"e\"]}]}" },
            "finish_reason": "length"
        }],
        "usage": { "prompt_tokens": prompt, "completion_tokens": completion }
    })
}

/// Measured 2026-09-24 (consolidate, session d3345285): 14,000 output tokens,
/// no call. A cut is sampled once more on the same key (the forced shape at
/// 0.9) and then handed back as `Truncated` for the lane chain to route; the
/// thinking is never offered for salvage and no text request follows.
#[tokio::test]
async fn no_tool_call_cut_at_the_limit_is_retried_once_then_truncated() {
    let (server, seen) = serve(vec![
        overthought_body(6_646, 4_000),
        overthought_body(6_646, 4_000),
        text_body("{\"summary\": \"s\", \"proposals\": []}"),
    ])
    .await;
    let err = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .complete_structured_raw(request(), schema())
        .await
        .expect_err("a cut is not a result");
    match err {
        ai_memory_llm::LlmError::Truncated {
            finish_reason,
            partial,
        } => {
            assert_eq!(finish_reason, "length");
            assert!(
                partial.is_none(),
                "reasoning is not output: never offered for salvage"
            );
        }
        other => panic!("expected Truncated, got {other:?}"),
    }
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "one same-key retry, no text fallback");
    assert!(seen.iter().all(|b| b.get("tools").is_some()));
}

/// Prose twice stays a shape error after exactly two forced requests.
#[tokio::test]
async fn prose_twice_is_a_shape_error_after_two_forced_requests() {
    let (server, seen) = serve(vec![text_body("prose"), text_body("still prose")]).await;
    let err = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .complete_structured_raw(request(), schema())
        .await
        .expect_err("no JSON");
    assert!(
        matches!(err, ai_memory_llm::LlmError::UnexpectedShape(_)),
        "{err:?}"
    );
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(seen.iter().all(|b| b.get("tools").is_some()));
}
