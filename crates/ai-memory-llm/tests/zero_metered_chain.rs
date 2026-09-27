//! The zero-metered lane chain end to end (FN8-9336): Laguna key A → key B
//! (transport failures only) → the ChatGPT seat through the Codex CLI's own
//! sign-in → no LLM. Laguna answers are streamed SSE; the seat is the real
//! `OpenAiOAuthProvider` reading a temp `auth.json` and answering from a
//! wiremock Responses endpoint.
//!
//! Hermetic: every ledger and alarm path is in a temp dir. The shared
//! actionable bus `/tmp/fn8-chyros-actionable.jsonl` must not grow while a
//! test body runs; each test checks its byte length around the body.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ai_memory_llm::types::{ChatMessage, ChatRequest, Role};
use ai_memory_llm::{
    DailyCap, FallbackProvider, JsonlLedger, Lane, LlmError, LlmProvider, OpenAiCompatProvider,
    OpenAiOAuthProvider, SystemWallClock, count_lane_calls_since, mark_chunkable, utc_day_start,
    with_caller,
};
use serde_json::{Value, json};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const BUS: &str = "/tmp/fn8-chyros-actionable.jsonl";

fn bus_len() -> Option<u64> {
    std::fs::metadata(BUS).ok().map(|m| m.len())
}

fn page_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "body_markdown": { "type": "string" },
            "tags": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["title", "body_markdown", "tags"],
        "additionalProperties": false
    })
}

fn page() -> Value {
    json!({ "title": "Session", "body_markdown": "## Done\n- shipped", "tags": ["laguna"] })
}

// ── SSE fixtures ──────────────────────────────────────────────────────────

fn event(delta: Value, finish: Option<&str>) -> String {
    let chunk = json!({
        "id": "c", "object": "chat.completion.chunk", "model": "poolside/laguna-s-2.1",
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }]
    });
    format!("data: {chunk}\n\n")
}

fn usage_event(prompt: u32, completion: u32) -> String {
    let chunk = json!({ "id": "c", "object": "chat.completion.chunk", "choices": [],
        "usage": { "prompt_tokens": prompt, "completion_tokens": completion } });
    format!("data: {chunk}\n\n")
}

fn sse(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(format!("{body}data: [DONE]\n\n"), "text/event-stream")
}

/// The measured loop shape: deliberation in `content` that never ends.
const LOOP: &str = "I'll analyze this session. Let me look at the observations again. \
Actually, I want to reconsider the approach before I call the function. ";
const LOOP_REPEATS: usize = 300;

/// A stream whose answer text holds a valid page and then loops.
fn looping_after_valid_json() -> ResponseTemplate {
    let mut body = event(json!({ "role": "assistant", "content": page().to_string() }), None);
    for _ in 0..LOOP_REPEATS {
        body.push_str(&event(json!({ "content": LOOP }), None));
    }
    body.push_str(&event(json!({}), Some("length")));
    body.push_str(&usage_event(7_000, 4_000));
    sse(body)
}

/// A stream that only loops.
fn looping_without_json() -> ResponseTemplate {
    let mut body = String::new();
    for _ in 0..LOOP_REPEATS {
        body.push_str(&event(json!({ "content": LOOP }), None));
    }
    body.push_str(&event(json!({}), Some("length")));
    body.push_str(&usage_event(7_000, 4_000));
    sse(body)
}

/// A clean forced call, arguments streamed in two pieces.
fn tool_call(args: &Value) -> ResponseTemplate {
    let text = args.to_string();
    let (a, b) = text.split_at(text.len() / 2);
    let mut body = event(
        json!({ "role": "assistant", "tool_calls": [{ "index": 0, "id": "t", "type": "function",
            "function": { "name": "submit_structured_output", "arguments": a } }] }),
        None,
    );
    body.push_str(&event(
        json!({ "tool_calls": [{ "index": 0, "function": { "arguments": b } }] }),
        None,
    ));
    body.push_str(&event(json!({}), Some("tool_calls")));
    body.push_str(&usage_event(7_000, 900));
    sse(body)
}

// ── Mock upstreams ────────────────────────────────────────────────────────

type Seen = Arc<Mutex<Vec<Value>>>;

#[derive(Clone)]
struct Queue {
    replies: Arc<Vec<ResponseTemplate>>,
    seen: Seen,
}

impl Respond for Queue {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let mut seen = self.seen.lock().unwrap();
        seen.push(serde_json::from_slice(&req.body).unwrap_or(Value::Null));
        let i = (seen.len() - 1).min(self.replies.len() - 1);
        self.replies[i].clone()
    }
}

async fn upstream(replies: Vec<ResponseTemplate>) -> (MockServer, Seen) {
    let server = MockServer::start().await;
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .respond_with(Queue {
            replies: Arc::new(replies),
            seen: seen.clone(),
        })
        .mount(&server)
        .await;
    (server, seen)
}

fn laguna(server: &MockServer, key: &str) -> Arc<dyn LlmProvider> {
    Arc::new(
        OpenAiCompatProvider::new(
            format!("{}/v1", server.uri()),
            Some(key.to_string().into()),
            "poolside/laguna-s-2.1",
        )
        .unwrap()
        // The mock host is not poolside.ai; force the toggle the real host gets.
        .with_disable_thinking(true),
    )
}

/// A Codex CLI `auth.json` whose access token expires in 2100.
fn codex_auth(dir: &std::path::Path) -> PathBuf {
    use base64::Engine as _;
    let b64 = |v: &Value| {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(v.to_string().as_bytes())
    };
    let jwt = format!(
        "{}.{}.sig",
        b64(&json!({ "alg": "none" })),
        b64(&json!({ "exp": 4_102_444_800_u64 }))
    );
    let path = dir.join("codex-auth.json");
    std::fs::write(
        &path,
        json!({ "auth_mode": "chatgpt", "tokens": {
            "access_token": jwt, "refresh_token": "fixture-refresh",
            "id_token": jwt, "account_id": "acct-fixture" } })
        .to_string(),
    )
    .unwrap();
    path
}

/// The seat's Responses stream carrying `answer` as output text.
fn seat_answer(answer: &Value) -> ResponseTemplate {
    let text = answer.to_string();
    let body = format!(
        "event: response.output_text.delta\ndata: {}\n\nevent: response.completed\ndata: {}\n\ndata: [DONE]\n\n",
        json!({ "type": "response.output_text.delta", "delta": text }),
        json!({ "type": "response.completed", "response": {
            "model": "gpt-6-astra", "usage": { "input_tokens": 7_100, "output_tokens": 800 },
            "output": [] } })
    );
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

fn seat(server: &MockServer, auth: PathBuf) -> Arc<dyn LlmProvider> {
    Arc::new(
        OpenAiOAuthProvider::from_codex_cli_auth(auth, "gpt-6-astra")
            .unwrap()
            .with_responses_url(format!("{}/codex/responses", server.uri())),
    )
}

struct Rig {
    chain: FallbackProvider,
    calls: PathBuf,
    _dir: tempfile::TempDir,
}

fn rig(
    key_a: Arc<dyn LlmProvider>,
    key_b: Arc<dyn LlmProvider>,
    seat: Arc<dyn LlmProvider>,
    cap: DailyCap,
    dir: tempfile::TempDir,
) -> Rig {
    let calls = dir.path().join("llm-calls.jsonl");
    let ledger = JsonlLedger::new(Some(calls.clone())).with_alarm(Some(dir.path().join("alarm.jsonl")));
    let chain = FallbackProvider::chain(
        Lane::new(key_a, "key-a").with_key_fp(Some("aaaa0000".into())),
        vec![
            Lane::new(key_b, "key-b")
                .with_key_fp(Some("bbbb1111".into()))
                .transport_only(),
            Lane::new(seat, "codex-oauth").with_daily_cap(cap),
        ],
    )
    .with_observer(Arc::new(ledger));
    Rig {
        chain,
        calls,
        _dir: dir,
    }
}

fn rows(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn consolidate_request(user: String) -> ChatRequest {
    ChatRequest {
        system: Some("You consolidate a coding session into one wiki page.".into()),
        messages: vec![ChatMessage {
            role: Role::User,
            content: user,
        }],
        max_tokens: 32_000,
        temperature: Some(0.2),
    }
}

async fn consolidate(chain: &FallbackProvider, request: ChatRequest) -> Result<Value, LlmError> {
    with_caller("consolidate", chain.complete_structured_raw(request, page_schema())).await
}

fn assert_contract(body: &Value, temperature: f64) {
    assert_eq!(body["chat_template_kwargs"], json!({ "enable_thinking": false }), "thinking off");
    assert_eq!(body["tool_choice"]["function"]["name"], json!("submit_structured_output"));
    assert_eq!(body["max_tokens"], json!(4_000));
    assert_eq!(body["stream"], json!(true));
    let t = body["temperature"].as_f64().unwrap();
    assert!((t - temperature).abs() < 1e-6, "temperature {t}, expected {temperature}");
}

// ── (a) ───────────────────────────────────────────────────────────────────

/// (a) A looping stream is aborted by the guard inside its window; the valid
/// page already in the answer text is salvaged; the consolidation is written
/// by key A with one request and no breaker opens.
#[tokio::test]
async fn a_loop_is_cut_short_and_the_valid_page_before_it_is_salvaged() {
    let before = bus_len();
    let (a_srv, a_seen) = upstream(vec![looping_after_valid_json()]).await;
    let (b_srv, b_seen) = upstream(vec![tool_call(&page())]).await;
    let (s_srv, s_seen) = upstream(vec![seat_answer(&page())]).await;
    let dir = tempfile::tempdir().unwrap();
    let auth = codex_auth(dir.path());
    let r = rig(laguna(&a_srv, "key-a-secret"), laguna(&b_srv, "key-b-secret"), seat(&s_srv, auth), DailyCap::new(40), dir);

    let out = consolidate(&r.chain, consolidate_request("observations".into())).await.unwrap();
    assert_eq!(out, page());
    assert_eq!(a_seen.lock().unwrap().len(), 1, "salvaged: no retry");
    assert_contract(&a_seen.lock().unwrap()[0], 0.7);
    assert_eq!((b_seen.lock().unwrap().len(), s_seen.lock().unwrap().len()), (0, 0));

    let rows = rows(&r.calls);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!((rows[0]["lane"].as_str(), rows[0]["outcome"].as_str()), (Some("key-a"), Some("ok")));
    assert!(rows.iter().all(|r| r["kind"] == "call"), "no breaker opened: {rows:?}");
    // The stream carried ~39k characters of loop; the guard stopped reading
    // within its window, so the (estimated) output booked is a small fraction.
    let full_stream_tokens = (LOOP.len() * LOOP_REPEATS / 4) as u64;
    let booked = rows[0]["output_tokens"].as_u64().unwrap();
    assert!(booked < 1_000, "booked {booked} output tokens of ~{full_stream_tokens}");
    assert_eq!(bus_len(), before, "the shared actionable bus was not written");
}

// ── (b) ───────────────────────────────────────────────────────────────────

/// (b) A loop with nothing to salvage: one more sample on key A at 0.9, then
/// the seat answers. Key B — a second key of the same model — is not called.
#[tokio::test]
async fn a_loop_without_a_page_retries_once_on_key_a_then_the_seat_answers() {
    let before = bus_len();
    let (a_srv, a_seen) = upstream(vec![looping_without_json(), looping_without_json()]).await;
    let (b_srv, b_seen) = upstream(vec![tool_call(&page())]).await;
    let (s_srv, s_seen) = upstream(vec![seat_answer(&page())]).await;
    let dir = tempfile::tempdir().unwrap();
    let auth = codex_auth(dir.path());
    let auth_bytes = std::fs::read(&auth).unwrap();
    let r = rig(laguna(&a_srv, "key-a-secret"), laguna(&b_srv, "key-b-secret"), seat(&s_srv, auth.clone()), DailyCap::new(40), dir);

    let out = consolidate(&r.chain, consolidate_request("observations".into())).await.unwrap();
    assert_eq!(out, page());
    {
        let a = a_seen.lock().unwrap();
        assert_eq!(a.len(), 2, "exactly one same-key retry and nothing else on key A");
        assert_contract(&a[0], 0.7);
        assert_contract(&a[1], 0.9);
    }
    assert_eq!(b_seen.lock().unwrap().len(), 0, "key B is for transport failures only");
    assert_eq!(s_seen.lock().unwrap().len(), 1);
    assert_eq!(std::fs::read(&auth).unwrap(), auth_bytes, "the Codex CLI's auth.json is never written");

    let rows = rows(&r.calls);
    let got: Vec<(&str, &str, Option<u64>)> = rows
        .iter()
        .map(|r| (r["lane"].as_str().unwrap(), r["outcome"].as_str().unwrap(), r["http_calls"].as_u64()))
        .collect();
    assert_eq!(got, vec![("key-a", "truncated", Some(2)), ("codex-oauth", "ok", Some(1))]);
    let seat_row = &rows[1];
    assert_eq!(seat_row["cost_usd_est"], json!(0.0));
    assert_eq!((seat_row["input_tokens"].as_u64(), seat_row["output_tokens"].as_u64()), (Some(7_100), Some(800)));
    assert!(seat_row["funding"].as_str().unwrap().contains("subscription flat"));
    assert_eq!(bus_len(), before);
}

// ── (c) ───────────────────────────────────────────────────────────────────

/// (c) A 429 on key A is a transport failure: key B gets the request.
#[tokio::test]
async fn a_429_on_key_a_goes_to_key_b() {
    let before = bus_len();
    let (a_srv, a_seen) = upstream(vec![ResponseTemplate::new(429).set_body_string("{\"error\":\"usage limit exceeded\"}")]).await;
    let (b_srv, b_seen) = upstream(vec![tool_call(&page())]).await;
    let (s_srv, s_seen) = upstream(vec![seat_answer(&page())]).await;
    let dir = tempfile::tempdir().unwrap();
    let auth = codex_auth(dir.path());
    let r = rig(laguna(&a_srv, "key-a-secret"), laguna(&b_srv, "key-b-secret"), seat(&s_srv, auth), DailyCap::new(40), dir);

    let (out, who) = ai_memory_llm::capture_responder(consolidate(&r.chain, consolidate_request("observations".into()))).await;
    assert_eq!(out.unwrap(), page());
    assert_eq!(who.unwrap().lane, "key-b");
    assert_eq!((a_seen.lock().unwrap().len(), b_seen.lock().unwrap().len(), s_seen.lock().unwrap().len()), (1, 1, 0));
    assert_contract(&b_seen.lock().unwrap()[0], 0.7);
    assert_eq!(bus_len(), before);
}

// ── (d) ───────────────────────────────────────────────────────────────────

/// Answers map calls (whose function schema has `facts`) with one fact per
/// chunk, and the reduce call with the page.
#[derive(Clone)]
struct MapReduce {
    seen: Seen,
}

impl Respond for MapReduce {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&req.body).unwrap();
        let is_map = body["tools"][0]["function"]["parameters"]["properties"]
            .get("facts")
            .is_some();
        let n = {
            let mut seen = self.seen.lock().unwrap();
            seen.push(body);
            seen.len()
        };
        if is_map {
            tool_call(&json!({ "facts": [format!("fact from part {n}")] }))
        } else {
            tool_call(&page())
        }
    }
}

/// (d) A 20k-token input is map-reduced: chunks of at most 8k tokens, one
/// reduce that carries the context outside the marked region verbatim, and one
/// ledger row for the whole consolidation.
#[tokio::test]
async fn a_20k_token_input_is_map_reduced_as_one_consolidation() {
    let before = bus_len();
    let a_srv = MockServer::start().await;
    let a_seen: Seen = Arc::new(Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .respond_with(MapReduce { seen: a_seen.clone() })
        .mount(&a_srv)
        .await;
    let (b_srv, b_seen) = upstream(vec![tool_call(&page())]).await;
    let (s_srv, s_seen) = upstream(vec![seat_answer(&page())]).await;
    let dir = tempfile::tempdir().unwrap();
    let auth = codex_auth(dir.path());
    let r = rig(laguna(&a_srv, "key-a-secret"), laguna(&b_srv, "key-b-secret"), seat(&s_srv, auth), DailyCap::new(40), dir);

    let observations: String = (0..600)
        .map(|i| format!("[{i:04}] tool_use Bash cargo test -p ai-memory-llm -> ok ({})\n", "x".repeat(60)))
        .collect();
    assert!(observations.len() >= 60_000, "~20k tokens at 3 chars/token");
    let head = "Session id: s-1\nCURRENT PAGE (keep verbatim): the old body.\n";
    let tail = "\nINSTRUCTIONS: follow the page conventions.\n";
    let user = format!("{head}{}{tail}", mark_chunkable(&observations));

    let out = consolidate(&r.chain, consolidate_request(user)).await.unwrap();
    assert_eq!(out, page());

    let seen = a_seen.lock().unwrap();
    let (maps, reduces): (Vec<&Value>, Vec<&Value>) = seen
        .iter()
        .partition(|b| b["tools"][0]["function"]["parameters"]["properties"].get("facts").is_some());
    assert_eq!(reduces.len(), 1, "one reduce");
    assert!((3..=4).contains(&maps.len()), "{} map calls", maps.len());
    for map in &maps {
        assert_contract(map, 0.7);
        let content = map["messages"].as_array().unwrap().last().unwrap()["content"].as_str().unwrap();
        let chunk = content.split_once("\n\n").unwrap().1;
        assert!(chunk.len() <= 8_000 * 3, "chunk of {} chars exceeds 8k tokens", chunk.len());
    }
    let reduce_user = reduces[0]["messages"].as_array().unwrap().last().unwrap()["content"]
        .as_str()
        .unwrap();
    assert!(reduce_user.starts_with(head), "context before the region reaches the reduce verbatim");
    assert!(reduce_user.ends_with(tail), "context after the region reaches the reduce verbatim");
    assert!(!reduce_user.contains("[0001] tool_use"), "the raw log is replaced by its facts");
    assert!(reduce_user.contains("fact from part 1"));
    assert!(!reduce_user.contains("ai-memory:chunkable"));
    assert_contract(reduces[0], 0.7);
    let n_maps = maps.len() as u64;
    drop(maps);
    drop(reduces);
    drop(seen);
    assert_eq!((b_seen.lock().unwrap().len(), s_seen.lock().unwrap().len()), (0, 0));

    let rows = rows(&r.calls);
    assert_eq!(rows.len(), 1, "one consolidation, one ledger row: {rows:?}");
    assert_eq!(rows[0]["caller"], json!("consolidate"));
    assert_eq!(rows[0]["http_calls"].as_u64(), Some(n_maps + 1));
    assert_eq!(bus_len(), before);
}

// ── (e) ───────────────────────────────────────────────────────────────────

/// (e) Request 41 of the UTC day on the seat: nothing is sent, and the chain
/// reports the allocation — not key A's loop — so the caller defers the job.
/// The 40 already used are read back from the ledger, as after a restart.
#[tokio::test]
async fn request_41_of_the_day_sends_nothing_and_reports_the_allocation() {
    let before = bus_len();
    let (a_srv, a_seen) = upstream(vec![looping_without_json(), looping_without_json()]).await;
    let (b_srv, b_seen) = upstream(vec![tool_call(&page())]).await;
    let (s_srv, s_seen) = upstream(vec![seat_answer(&page())]).await;
    let dir = tempfile::tempdir().unwrap();
    let auth = codex_auth(dir.path());
    let calls = dir.path().join("llm-calls.jsonl");
    let now = jiff::Timestamp::now().to_string();
    let mut prior = String::new();
    for _ in 0..40 {
        prior.push_str(&format!("{}\n", json!({ "ts": now, "kind": "call", "lane": "codex-oauth", "outcome": "ok" })));
    }
    std::fs::write(&calls, prior).unwrap();
    let today = utc_day_start(jiff::Timestamp::now().as_second());
    let used = count_lane_calls_since(&calls, "codex-oauth", today);
    assert_eq!(used, 40);
    let cap = DailyCap::with_wall_clock(40, Arc::new(SystemWallClock), used);
    let r = rig(laguna(&a_srv, "key-a-secret"), laguna(&b_srv, "key-b-secret"), seat(&s_srv, auth), cap, dir);

    match consolidate(&r.chain, consolidate_request("observations".into())).await {
        Err(LlmError::AllocationExhausted { lane, used, cap, resets_at_unix }) => {
            assert_eq!((lane.as_str(), used, cap), ("codex-oauth", 40, 40));
            assert_eq!(resets_at_unix, today + 86_400);
        }
        other => panic!("expected AllocationExhausted, got {other:?}"),
    }
    assert_eq!(s_seen.lock().unwrap().len(), 0, "zero HTTP to the seat");
    assert_eq!(b_seen.lock().unwrap().len(), 0);
    assert_eq!(a_seen.lock().unwrap().len(), 2);
    let rows = rows(&r.calls);
    let last = rows.last().unwrap();
    assert_eq!((last["lane"].as_str(), last["outcome"].as_str()), (Some("codex-oauth"), Some("capped")));
    assert_eq!(count_lane_calls_since(&r.calls, "codex-oauth", today), 40, "a refusal is not a spent request");
    assert_eq!(bus_len(), before);
}

// ── (f) ───────────────────────────────────────────────────────────────────

/// (f) Only Laguna on inference.poolside.ai and the ChatGPT seat are allowed
/// in the chain; the same openai-compat wire pointed at a metered gateway is
/// not, whatever the model is called.
#[test]
fn only_laguna_on_poolside_and_the_seat_are_zero_metered() {
    use ai_memory_llm::{LaneKind, lane_kind};
    let laguna = "poolside/laguna-s-2.1";
    assert_eq!(lane_kind("openai-compat", laguna, Some("https://inference.poolside.ai/v1")), LaneKind::Laguna);
    assert_eq!(lane_kind("openai-oauth", "gpt-6-astra", None), LaneKind::CodexSeat);
    for (provider, model, endpoint) in [
        ("openai-compat", laguna, Some("https://openrouter.ai/api/v1")),
        ("openai-compat", laguna, Some("https://opencode.ai/zen/v1")),
        ("openai-compat", laguna, Some("https://inference.poolside.ai.evil.example/v1")),
        ("openai-compat", laguna, None),
        ("openai-compat", "qwen3", Some("https://inference.poolside.ai/v1")),
        ("gemini", "gemini-2.5-flash", None),
        ("openai", "gpt-5.5", Some("https://api.openai.com")),
        ("anthropic", "claude-sonnet-5", None),
        ("opencode", "kimi-k3", None),
    ] {
        assert!(!lane_kind(provider, model, endpoint).is_zero_metered(), "{provider} {model} {endpoint:?}");
    }
}
