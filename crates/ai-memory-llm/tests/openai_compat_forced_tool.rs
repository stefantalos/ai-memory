//! Forced-tool structured output for Poolside/Laguna, against a wiremock
//! upstream.
//!
//! Measured live 2026-09-23 on inference.poolside.ai with
//! `poolside/laguna-s-2.1`: `response_format` with a strict json_schema was
//! ignored 4 of 4 times, while one forced function call with
//! `chat_template_kwargs.enable_thinking=false` returned a schema-conformant
//! object 9 of 11 times; the other two carried a nested array as a
//! JSON-encoded string. These tests pin the request shape and the parsing.

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

#[tokio::test]
async fn no_tool_call_retries_with_stricter_instruction() {
    let (server, seen) = serve(vec![
        text_body("I'll propose two pages."),
        text_body("here: {\"summary\": \"s\", \"proposals\": []}"),
    ])
    .await;
    let out = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .with_stricter_retry_on_shape_fault(true)
        .complete_structured_raw(request(), schema())
        .await
        .expect("fallback result");
    assert_eq!(out, json!({ "summary": "s", "proposals": [] }));
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(
        seen[1].get("tools").is_some(),
        "the retry is a structured call"
    );
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
async fn poolside_output_is_capped_at_14k_on_every_call() {
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
    assert_eq!(
        seen[0]["max_tokens"],
        json!(14_000),
        "structured call capped"
    );
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
/// no call. Before the fix this was a shape error, so the SAME prompt was sent
/// again as a text call with thinking on and its failure masked the cut.
#[tokio::test]
async fn no_tool_call_cut_at_the_limit_is_truncated_and_sends_nothing_else() {
    let (server, seen) = serve(vec![
        overthought_body(6_646, 14_000),
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
    assert_eq!(
        seen.lock().unwrap().len(),
        1,
        "no text fallback on the same lane after a cut"
    );
}

/// The text fallback still runs when the forced call ended for another
/// reason — but if THAT call is cut at the limit with no JSON, the caller must
/// see `Truncated` (routable to the next lane), not a fatal shape error.
#[tokio::test]
async fn a_text_fallback_cut_at_the_limit_is_truncated_not_a_shape_error() {
    let cut_text = json!({
        "id": "id", "object": "chat.completion", "created": 0, "model": "poolside/laguna-s-2.1",
        "choices": [{ "index": 0,
            "message": { "role": "assistant", "content": "<think>first I will list every" },
            "finish_reason": "length" }]
    });
    let (server, seen) = serve(vec![text_body("I'll propose two pages."), cut_text]).await;
    let err = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .with_stricter_retry_on_shape_fault(true)
        .complete_structured_raw(request(), schema())
        .await
        .expect_err("a cut is not a result");
    assert!(
        matches!(err, ai_memory_llm::LlmError::Truncated { .. }),
        "expected Truncated, got {err:?}"
    );
    assert_eq!(seen.lock().unwrap().len(), 2);
}

/// Same prose without the cut stays a shape error: the classification rides
/// on `finish_reason`, not on the absence of JSON alone.
#[tokio::test]
async fn a_text_fallback_without_json_that_stopped_normally_stays_a_shape_error() {
    let (server, _) = serve(vec![text_body("prose"), text_body("still prose")]).await;
    let err = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .complete_structured_raw(request(), schema())
        .await
        .expect_err("no JSON");
    assert!(
        matches!(err, ai_memory_llm::LlmError::UnexpectedShape(_)),
        "{err:?}"
    );
}

/// Measured 2026-09-24 on b50cd18b: three consolidate calls each booked
/// http_calls=2 and ~16.6k output tokens: a forced call that ended in prose
/// without the function, then this fallback as a plain text call with
/// thinking at the template default, cut at 14,000 every time. For Poolside
/// the prose reply is the answer: a shape error, and nothing else is sent.
#[tokio::test]
async fn poolside_prose_instead_of_the_call_retries_and_salvages_json() {
    let (server, seen) = serve(vec![
        text_body("Let me analyze this session carefully. Actually, I want to reconsider"),
        text_body("{\"summary\": \"s\", \"proposals\": []}"),
    ])
    .await;
    let out = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1")
        .complete_structured_raw(request(), schema())
        .await
        .expect("prose with JSON on retry is salvaged");
    assert_eq!(out, json!({ "summary": "s", "proposals": [] }));
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "second strict retry request was made");
    
    // Check that the second request had the strict system instruction
    let messages = seen[1]["messages"].as_array().unwrap();
    let sys = messages[0]["content"].as_str().unwrap();
    assert!(sys.contains("CRITICAL: You MUST call the provided tool to submit your answer. Do NOT return prose."));
}

#[tokio::test]
async fn other_models_on_the_forced_tool_path_keep_the_text_fallback() {
    let (server, seen) = serve(vec![
        text_body("prose"),
        text_body("{\"summary\": \"s\", \"proposals\": []}"),
    ])
    .await;
    let out = provider(&format!("{}/v1", server.uri()), "mistral-nemo")
        .with_structured_via_tool(true)
        .complete_structured_raw(request(), schema())
        .await
        .expect("fallback result");
    assert_eq!(out, json!({ "summary": "s", "proposals": [] }));
    assert_eq!(seen.lock().unwrap().len(), 2);
}

/// A/B 2026-09-25 (docs/research/laguna-loop-ab-2026-09-25.md): at the
/// callers' 0.1 Laguna was cut or lost 5 of 7 times on the looping inputs, at
/// 0.7 2 of 8. Every Poolside request carries 0.7, whatever the caller set.
#[tokio::test]
async fn poolside_requests_carry_the_poolside_temperature_on_every_path() {
    let (server, seen) = serve(vec![
        tool_call_body(json!({ "summary": "s", "proposals": [] })),
        text_body("plain"),
    ])
    .await;
    let p = provider(&format!("{}/v1", server.uri()), "poolside/laguna-s-2.1");
    let mut r = request();
    r.temperature = Some(0.1);
    p.complete_structured_raw(r.clone(), schema())
        .await
        .unwrap();
    p.complete(r).await.unwrap();
    let seen = seen.lock().unwrap();
    let t0 = seen[0]["temperature"].as_f64().expect("temperature sent");
    let t1 = seen[1]["temperature"].as_f64().expect("temperature sent");
    assert!((t0 - 0.7).abs() < 1e-6, "forced call: {t0}");
    assert!((t1 - 0.7).abs() < 1e-6, "text call: {t1}");
}

#[tokio::test]
async fn other_models_keep_the_callers_temperature() {
    let (server, seen) = serve(vec![text_body("{\"summary\": \"s\", \"proposals\": []}")]).await;
    let mut r = request();
    r.temperature = Some(0.1);
    provider(&format!("{}/v1", server.uri()), "mistral-nemo")
        .complete_structured_raw(r, schema())
        .await
        .unwrap();
    let t = seen.lock().unwrap()[0]["temperature"]
        .as_f64()
        .expect("temperature sent");
    assert!((t - 0.1).abs() < 1e-6, "{t}");
}
