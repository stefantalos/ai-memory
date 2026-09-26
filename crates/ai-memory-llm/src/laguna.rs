//! The Laguna contract: how ai-memory asks `poolside/laguna-*` for structured
//! output (FN8-9336).
//!
//! ref: docs/research/laguna-loop-ab-2026-09-25.md (the A/B these values come
//! from) · crates/ai-memory-llm/src/fallback.rs (what happens after a miss).
//!
//! Measured on Laguna S 2.1 (A/B 2026-09-25, 28 calls): thinking has only two
//! settings, off and max; with thinking on the answer loops every time (cut
//! 4/4); with thinking off the model sometimes "thinks aloud" in `content` and
//! loops there instead, never calling the function; at temperature 0.7 the
//! forced call was cut 0/4 on the input that looped at 0.1, 2/8 pooled; and
//! every answer that finished, in every variant, passed ai-memory's own
//! validation. The failures are loops running into the output ceiling, not
//! bad JSON. So the contract is:
//!
//! * thinking off, forced `tool_choice`, temperature [`LAGUNA_TEMPERATURE`],
//!   output ceiling [`LAGUNA_MAX_OUTPUT_TOKENS`];
//! * the reply is streamed and a [`RepetitionGuard`] aborts it as soon as the
//!   tail of `content` (or of the thinking) repeats itself — a loop is
//!   recognised in seconds instead of being paid for up to the ceiling;
//! * salvage: when no usable call came back but `content` already holds a JSON
//!   object that passes the strict validator ([`validate`]), it is accepted;
//! * otherwise one retry on the same key at [`LAGUNA_RETRY_TEMPERATURE`]; then
//!   the error goes back to the lane chain, which moves on to the next
//!   non-transport-only lane. There is no plain-text second request with
//!   thinking at the template default: that request shape was cut 4 of 4 times.
//! * inputs above [`MAP_REDUCE_THRESHOLD_TOKENS`] are map-reduced: the region
//!   the caller marked with [`mark_chunkable`] is split into chunks of at most
//!   [`MAP_CHUNK_MAX_TOKENS`], each chunk is read by one map call that returns
//!   partial facts, and one reduce call — the caller's own prompt with the
//!   marked region replaced by those facts — produces the answer. Long inputs
//!   (24–33k tokens) failed 5 of 7 in the A/B.

use serde_json::Value;

use crate::error::{LlmError, LlmResult};
use crate::openai::{OpenAiProvider, StreamOutcome, decode_stringified_containers};
use crate::openai_compat::{first_json_object, strip_reasoning_blocks};
use crate::types::{ChatMessage, ChatRequest, LlmOperationId, Role};

/// Sampling temperature of the first attempt (Poolside's own vLLM recipe;
/// A/B: cut 0/4 on the input that was cut 2/4 at 0.1).
pub const LAGUNA_TEMPERATURE: f32 = 0.7;

/// Sampling temperature of the one same-key retry: a different sample of the
/// same request. `seed` is not sent: Poolside's support for it is unverified,
/// and a 400 on an unknown parameter would be a deterministic failure.
pub const LAGUNA_RETRY_TEMPERATURE: f32 = 0.9;

/// Output ceiling for every Laguna structured call. The A/B's finished
/// consolidations used 827–1,120 output tokens; the 14,000 ceiling it replaces
/// only ever paid for loops. Finished auto_improve answers ran to 5,944 (4 of
/// 12 above 4,000): those are now cut here and answered by the next lane.
pub const LAGUNA_MAX_OUTPUT_TOKENS: u32 = 4_000;

/// Estimated prompt size above which the marked region is map-reduced.
pub const MAP_REDUCE_THRESHOLD_TOKENS: usize = 12_000;

/// Largest chunk one map call reads.
pub const MAP_CHUNK_MAX_TOKENS: usize = 8_000;

/// Characters per token for these estimates: the same conservative figure the
/// consolidation prompt budgets use (`CHARS_PER_TOKEN` in ai-memory-consolidate).
pub const CHARS_PER_TOKEN: usize = 3;

/// Tail of the answer the repetition guard inspects.
pub const GUARD_WINDOW_CHARS: usize = 1_500;

/// Words in one n-gram window of the guard.
pub const GUARD_NGRAM_WORDS: usize = 12;

/// Occurrences of one n-gram inside the window that mean "looping".
pub const GUARD_MIN_REPEATS: usize = 3;

/// The guard does not judge an answer shorter than this.
pub const GUARD_MIN_CHARS: usize = GUARD_WINDOW_CHARS;

/// How often (in newly streamed characters) the guard looks again.
pub const GUARD_CHECK_EVERY_CHARS: usize = 256;

const CHUNK_BEGIN: &str = "<!-- ai-memory:chunkable:begin -->\n";
const CHUNK_END: &str = "\n<!-- ai-memory:chunkable:end -->";

/// Characters [`mark_chunkable`] adds around a region; prompt budgets that
/// wrap a region subtract it so the marked prompt still fits.
pub const CHUNKABLE_MARKER_CHARS: usize = CHUNK_BEGIN.len() + CHUNK_END.len();

/// Wrap the part of a prompt that may be read in pieces (the observation log)
/// so a provider with a small working budget can map-reduce it. Everything
/// outside the markers — instructions, the current page, patchable targets —
/// reaches the final call verbatim.
#[must_use]
pub fn mark_chunkable(text: &str) -> String {
    format!("{CHUNK_BEGIN}{text}{CHUNK_END}")
}

/// Detects a model repeating itself at the end of a streamed answer.
///
/// Measured loop shape (A/B 2026-09-25): 59–69k characters of deliberation
/// with a zlib ratio of 0.06–0.12, i.e. the same sentences over and over. The
/// raw loop bodies were not kept, so the window is chosen, not fitted: in the
/// last [`GUARD_WINDOW_CHARS`] characters, some run of [`GUARD_NGRAM_WORDS`]
/// words occurring [`GUARD_MIN_REPEATS`] times. Prose that is merely similar
/// (a list whose items share a prefix) differs somewhere in every twelve-word
/// run and does not trip it. Tool-call arguments are never judged: JSON
/// legitimately repeats its keys.
#[derive(Debug, Clone, Copy)]
pub struct RepetitionGuard {
    window_chars: usize,
    ngram_words: usize,
    min_repeats: usize,
    min_chars: usize,
}

impl Default for RepetitionGuard {
    fn default() -> Self {
        Self {
            window_chars: GUARD_WINDOW_CHARS,
            ngram_words: GUARD_NGRAM_WORDS,
            min_repeats: GUARD_MIN_REPEATS,
            min_chars: GUARD_MIN_CHARS,
        }
    }
}

impl RepetitionGuard {
    /// Whether the tail of `text` is a loop.
    #[must_use]
    pub fn is_looping(&self, text: &str) -> bool {
        if text.len() < self.min_chars {
            return false;
        }
        let mut start = text.len().saturating_sub(self.window_chars);
        while !text.is_char_boundary(start) {
            start += 1;
        }
        let words: Vec<String> = text[start..]
            .split_whitespace()
            .map(str::to_lowercase)
            .collect();
        if words.len() < self.ngram_words * self.min_repeats {
            return false;
        }
        let mut seen: std::collections::HashMap<&[String], usize> =
            std::collections::HashMap::new();
        for window in words.windows(self.ngram_words) {
            let n = seen.entry(window).or_insert(0);
            *n += 1;
            if *n >= self.min_repeats {
                return true;
            }
        }
        false
    }
}

/// Strict structural validation of `value` against a JSON schema as schemars
/// emits it: `$ref`, `type` (including `null` unions), `required`,
/// `properties`, `additionalProperties: false`, `items`, `enum`, `const`,
/// `anyOf` / `oneOf` / `allOf`. At least as strict as the callers' typed
/// deserialisation on everything it checks, so a salvaged answer that passes
/// here does not fail later outside the lane chain.
///
/// # Errors
/// A message naming the first offending path.
pub fn validate(value: &Value, schema: &Value) -> Result<(), String> {
    validate_at(value, schema, schema, "$")
}

fn resolve<'a>(node: &'a Value, root: &'a Value) -> Result<&'a Value, String> {
    let Some(reference) = node.get("$ref").and_then(Value::as_str) else {
        return Ok(node);
    };
    let path = reference
        .strip_prefix("#/")
        .ok_or_else(|| format!("unsupported $ref {reference}"))?;
    let mut cur = root;
    for seg in path.split('/') {
        cur = cur
            .get(seg)
            .ok_or_else(|| format!("unresolved $ref {reference}"))?;
    }
    resolve(cur, root)
}

fn type_matches(value: &Value, ty: &str) -> bool {
    match ty {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        _ => true,
    }
}

fn validate_at(value: &Value, node: &Value, root: &Value, at: &str) -> Result<(), String> {
    let node = resolve(node, root)?;
    if node.as_bool() == Some(true) || node.as_object().is_some_and(serde_json::Map::is_empty) {
        return Ok(());
    }
    if node.as_bool() == Some(false) {
        return Err(format!("{at}: not allowed"));
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = node.get(key).and_then(Value::as_array)
            && !branches
                .iter()
                .any(|b| validate_at(value, b, root, at).is_ok())
        {
            return Err(format!("{at}: matches no {key} branch"));
        }
    }
    if let Some(all) = node.get("allOf").and_then(Value::as_array) {
        for b in all {
            validate_at(value, b, root, at)?;
        }
    }
    match node.get("type") {
        Some(Value::String(ty)) if !type_matches(value, ty) => {
            return Err(format!("{at}: expected {ty}"));
        }
        Some(Value::Array(types))
            if !types
                .iter()
                .filter_map(Value::as_str)
                .any(|ty| type_matches(value, ty)) =>
        {
            return Err(format!("{at}: expected one of {types:?}"));
        }
        _ => {}
    }
    if let Some(options) = node.get("enum").and_then(Value::as_array)
        && !options.contains(value)
    {
        return Err(format!("{at}: not one of the allowed values"));
    }
    if let Some(expected) = node.get("const")
        && expected != value
    {
        return Err(format!("{at}: expected {expected}"));
    }
    if let Some(obj) = value.as_object() {
        if let Some(required) = node.get("required").and_then(Value::as_array) {
            for key in required.iter().filter_map(Value::as_str) {
                if !obj.contains_key(key) {
                    return Err(format!("{at}: missing required `{key}`"));
                }
            }
        }
        let props = node.get("properties").and_then(Value::as_object);
        for (key, child) in obj {
            match props.and_then(|p| p.get(key)) {
                Some(child_schema) => {
                    validate_at(child, child_schema, root, &format!("{at}.{key}"))?;
                }
                None => match node.get("additionalProperties") {
                    Some(Value::Bool(false)) => {
                        return Err(format!("{at}: unknown field `{key}`"));
                    }
                    Some(extra) if extra.is_object() => {
                        validate_at(child, extra, root, &format!("{at}.{key}"))?;
                    }
                    _ => {}
                },
            }
        }
    }
    if let (Some(items), Some(arr)) = (node.get("items"), value.as_array()) {
        for (i, item) in arr.iter().enumerate() {
            validate_at(item, items, root, &format!("{at}[{i}]"))?;
        }
    }
    Ok(())
}

/// Parse, repair and validate one candidate object.
fn accept(mut value: Value, schema: &Value) -> LlmResult<Value> {
    if !value.is_object() {
        return Err(LlmError::UnexpectedShape(
            "structured output is not a JSON object".into(),
        ));
    }
    decode_stringified_containers(&mut value, schema);
    validate(&value, schema)
        .map_err(|e| LlmError::UnexpectedShape(format!("structured output failed validation: {e}")))?;
    Ok(value)
}

/// A JSON object already sitting in the answer text, if it passes [`validate`].
fn salvage(content: &str, schema: &Value) -> Option<Value> {
    let cleaned = strip_reasoning_blocks(content);
    let candidate = serde_json::from_str::<Value>(&cleaned)
        .ok()
        .filter(Value::is_object)
        .or_else(|| {
            first_json_object(&cleaned).and_then(|s| serde_json::from_str::<Value>(s).ok())
        })?;
    accept(candidate, schema).ok()
}

/// Turn one streamed reply into the answer or the reason there is none.
fn judge(out: StreamOutcome, schema: &Value) -> LlmResult<Value> {
    let cut = out.finish_reason.as_deref() == Some("length");
    let mut call_error = None;
    if out.tool_called && !out.aborted {
        let parsed = match serde_json::from_str::<Value>(&out.arguments) {
            Ok(v) => accept(v, schema),
            Err(err) => Err(LlmError::from(err)),
        };
        match parsed {
            Ok(v) => return Ok(v),
            Err(err) => call_error = Some(err),
        }
    }
    if let Some(v) = salvage(&out.content, schema) {
        tracing::info!("Laguna contract: accepted a valid JSON object salvaged from the answer text");
        return Ok(v);
    }
    if out.aborted {
        return Err(LlmError::Truncated {
            finish_reason: "repetition".into(),
            partial: None,
        });
    }
    if cut {
        let partial = if out.tool_called {
            out.arguments
        } else {
            out.content
        };
        return Err(LlmError::Truncated {
            finish_reason: "length".into(),
            partial: (!partial.trim().is_empty()).then_some(crate::error::PartialText(partial)),
        });
    }
    Err(call_error.unwrap_or_else(|| {
        LlmError::UnexpectedShape("model did not call the forced structured-output function".into())
    }))
}

/// Whether a failed attempt is worth one more sample on the same key.
fn worth_a_second_sample(err: &LlmError) -> bool {
    matches!(
        err,
        LlmError::Truncated { .. }
            | LlmError::UnexpectedShape(_)
            | LlmError::Serde(_)
            | LlmError::EmptyResponse(_)
    )
}

/// One request under the contract: first sample, then at most one more.
async fn sampled(
    inner: &OpenAiProvider,
    request: &ChatRequest,
    schema: &Value,
    operation_id: LlmOperationId,
) -> LlmResult<Value> {
    let guard = RepetitionGuard::default();
    let mut first = request.clone();
    first.temperature = Some(LAGUNA_TEMPERATURE);
    first.max_tokens = first.max_tokens.min(LAGUNA_MAX_OUTPUT_TOKENS);
    let err = match inner
        .stream_forced_tool(&first, schema, operation_id, &guard)
        .await
        .and_then(|out| judge(out, schema))
    {
        Ok(v) => return Ok(v),
        Err(err) if worth_a_second_sample(&err) => err,
        Err(err) => return Err(err),
    };
    tracing::warn!(error = %err, "Laguna contract: first sample failed; one retry on the same key at a different temperature");
    let mut second = first;
    second.temperature = Some(LAGUNA_RETRY_TEMPERATURE);
    inner
        .stream_forced_tool(&second, schema, operation_id, &guard)
        .await
        .and_then(|out| judge(out, schema))
}

fn estimate_tokens(request: &ChatRequest) -> usize {
    let chars = request.system.as_deref().map_or(0, str::len)
        + request.messages.iter().map(|m| m.content.len()).sum::<usize>();
    chars / CHARS_PER_TOKEN
}

/// `(message index, region start, region end)` of the marked region, where
/// start/end bound the region's text (markers excluded).
fn find_region(request: &ChatRequest) -> Option<(usize, usize, usize)> {
    request.messages.iter().enumerate().find_map(|(i, m)| {
        let begin = m.content.find(CHUNK_BEGIN)?;
        let start = begin + CHUNK_BEGIN.len();
        let end = start + m.content[start..].find(CHUNK_END)?;
        Some((i, start, end))
    })
}

/// Split `text` into pieces of at most `max_chars` bytes, on line boundaries
/// where possible.
fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in text.split_inclusive('\n') {
        let mut line = line;
        while !line.is_empty() {
            let room = max_chars - cur.len();
            if line.len() <= room {
                cur.push_str(line);
                break;
            }
            if !cur.is_empty() {
                chunks.push(std::mem::take(&mut cur));
                continue;
            }
            let mut cut = max_chars.min(line.len());
            while !line.is_char_boundary(cut) {
                cut -= 1;
            }
            chunks.push(line[..cut].to_string());
            line = &line[cut..];
        }
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    chunks
}

fn map_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Every fact, decision, file, command, error and outcome in this part that the final task needs, one short self-contained sentence each."
            }
        },
        "required": ["facts"],
        "additionalProperties": false
    })
}

const MAP_SYSTEM: &str = "You read one part of a long evidence log and extract the facts a later step \
needs. Do not answer the final task. Call the function once with the facts of this part: short, \
self-contained sentences that keep names, paths, numbers, errors and decisions exactly as written. \
No analysis, no prose outside the function call.";

/// Structured completion under the Laguna contract (see module docs).
///
/// # Errors
/// Transport and provider errors unchanged (the lane chain decides where the
/// request goes next); after the same-key retry, the last content failure.
pub(crate) async fn complete_structured(
    inner: &OpenAiProvider,
    request: ChatRequest,
    schema: Value,
    operation_id: LlmOperationId,
) -> LlmResult<Value> {
    let region = find_region(&request);
    let Some((index, start, end)) =
        region.filter(|_| estimate_tokens(&request) > MAP_REDUCE_THRESHOLD_TOKENS)
    else {
        return sampled(inner, &request, &schema, operation_id).await;
    };
    let original = &request.messages[index].content;
    let chunks = chunk_text(&original[start..end], MAP_CHUNK_MAX_TOKENS * CHARS_PER_TOKEN);
    let task_context: String = request
        .system
        .as_deref()
        .unwrap_or("")
        .chars()
        .take(2_000)
        .collect();
    let total = chunks.len();
    tracing::info!(
        estimated_tokens = estimate_tokens(&request),
        chunks = total,
        "Laguna contract: long input, map-reduce over the marked region",
    );
    let mut facts: Vec<String> = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let map = ChatRequest {
            system: Some(format!(
                "{MAP_SYSTEM}\n\nThe final task, for context only:\n{task_context}"
            )),
            messages: vec![ChatMessage {
                role: Role::User,
                content: format!("Part {} of {total} of the evidence log:\n\n{chunk}", i + 1),
            }],
            max_tokens: LAGUNA_MAX_OUTPUT_TOKENS,
            temperature: Some(LAGUNA_TEMPERATURE),
        };
        let partial = sampled(inner, &map, &map_schema(), operation_id).await?;
        facts.extend(
            partial["facts"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string),
        );
    }
    let mut rendered = format!(
        "(The evidence log was {total} parts long; these are the facts extracted from it, in order.)\n"
    );
    for fact in &facts {
        rendered.push_str("- ");
        rendered.push_str(fact);
        rendered.push('\n');
    }
    let mut reduce = request.clone();
    let begin = start - CHUNK_BEGIN.len();
    let finish = end + CHUNK_END.len();
    reduce.messages[index].content = format!(
        "{}{}{}",
        &original[..begin],
        rendered,
        &original[finish..]
    );
    sampled(inner, &reduce, &schema, operation_id).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn looping_text() -> String {
        "I'll analyze this session. Let me look at the observations again. Actually, I want to \
         reconsider the approach before I call the function. "
            .repeat(40)
    }

    #[test]
    fn the_guard_trips_on_a_loop() {
        assert!(RepetitionGuard::default().is_looping(&looping_text()));
    }

    #[test]
    fn the_guard_does_not_judge_a_short_answer() {
        assert!(!RepetitionGuard::default().is_looping(&looping_text()[..1_000]));
    }

    #[test]
    fn a_long_page_with_a_repetitive_list_does_not_trip_the_guard() {
        let mut page = String::from("## Decisions\n\n");
        for i in 0..80 {
            page.push_str(&format!(
                "- Decision {i}: the consolidation worker now defers job {i} to the reset \
                 because lane {} reported cap {} at {}Z.\n",
                i % 3,
                40 + i,
                i * 7
            ));
        }
        assert!(page.len() > 3 * GUARD_WINDOW_CHARS);
        assert!(!RepetitionGuard::default().is_looping(&page));
    }

    #[test]
    fn validation_follows_refs_required_types_and_enums() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["title", "kind", "items"],
            "properties": {
                "title": {"type": "string"},
                "kind": {"$ref": "#/$defs/Kind"},
                "items": {"type": "array", "items": {"type": "integer"}},
                "note": {"type": ["string", "null"]}
            },
            "additionalProperties": false,
            "$defs": {"Kind": {"type": "string", "enum": ["fact", "rule"]}}
        });
        let ok = serde_json::json!({"title": "t", "kind": "fact", "items": [1, 2], "note": null});
        assert!(validate(&ok, &schema).is_ok());
        for bad in [
            serde_json::json!({"title": "t", "kind": "fact"}),
            serde_json::json!({"title": 1, "kind": "fact", "items": []}),
            serde_json::json!({"title": "t", "kind": "opinion", "items": []}),
            serde_json::json!({"title": "t", "kind": "fact", "items": ["x"]}),
            serde_json::json!({"title": "t", "kind": "fact", "items": [], "extra": 1}),
        ] {
            assert!(validate(&bad, &schema).is_err(), "{bad}");
        }
    }

    #[test]
    fn chunks_respect_the_limit_and_keep_every_byte() {
        let text: String = (0..500).map(|i| format!("line {i} {}\n", "x".repeat(i % 90))).collect();
        let chunks = chunk_text(&text, 4_000);
        assert!(chunks.iter().all(|c| c.len() <= 4_000));
        assert_eq!(chunks.concat(), text);
        let long = "y".repeat(10_000);
        let chunks = chunk_text(&long, 4_000);
        assert_eq!(chunks.iter().map(String::len).collect::<Vec<_>>(), vec![4_000, 4_000, 2_000]);
    }

    #[test]
    fn a_region_is_found_between_the_markers() {
        let req = ChatRequest::user_prompt(format!("head\n{}\ntail", mark_chunkable("OBS")));
        let (i, s, e) = find_region(&req).unwrap();
        assert_eq!(&req.messages[i].content[s..e], "OBS");
    }
}
