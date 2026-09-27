# Laguna S 2.1 output loop: A/B, 2026-09-24/25

Refs: FN8-8120. ref: crates/ai-memory-llm/src/openai.rs:434 (complete_structured_via_tool),
crates/ai-memory-llm/src/openai_compat.rs (complete_structured).

Raw rows: `laguna-loop-ab-2026-09-25.jsonl` (one per provider call) and
`laguna-loop-ab-2026-09-25.replay.jsonl` (ai-memory's own verdict on each finished answer).

## Method

- **Requests.** The exact request bodies were captured from binary `b50cd18b` (HEAD c09b061),
  not rebuilt by hand. A scratch `ai-memory serve` ran on an APFS copy-on-write snapshot of
  `~/.fn8/ai-memory`. Its `LLM_BASE_URL` pointed at a local stub (the path contains
  `poolside.ai`, so the thinking toggle is sent), and it used a dummy key. Scheduler and eval
  were off. Captured:
  - auto_improve `c80b0238`: 24,433 prompt tokens, cut at 20:11Z;
  - auto_improve `322b5b80`: 33,550 prompt tokens, cut at 20:19Z;
  - consolidate `ede4b765`: 7,762 prompt tokens, the 6,646 → 14,000 case at 18:42Z.
- **Shape of the calls.** Each call is the forced-tool request: `tools` plus a forced
  `tool_choice`, `chat_template_kwargs.enable_thinking=false`, `max_tokens` 14,000, and
  temperature 0.1 (auto_improve) or 0.2 (consolidate). When no tool call comes back, the
  binary sends a second, plain text request: no tools, no kwargs (thinking at the template
  default), `max_tokens` 14,000.
- **Endpoint and keys.** `https://inference.poolside.ai/v1`, key B only. A 16-token probe on
  key A returned `429 {"error":"usage limit exceeded"}` with no reset header.
- **Budget.** 28 calls on key B, run sequentially, no 429. This includes one unintended
  production call, disclosed below.
- **Validity check.** Each answer that finished was served back through a second scratch
  instance (stub in replay mode), and ai-memory's own parser and validator judged it.
- **Disclosure.** The first capture attempt ran the CLI without `AI_MEMORY_SERVER_URL`. It
  reached the production daemon and ran one real auto_improve on `c80b0238` at 21:19:17Z
  (key B, 24,453 → 1,022 tokens, ok, auto-approved). It is counted in the 28.

## Variants

| id | change vs V0 |
|---|---|
| V0 | none: what the binary sends today |
| T0.7 | temperature 0.7 |
| Von | `chat_template_kwargs` removed, so thinking runs at the template default (max), with tools kept. This is the thinking regime of the text fallback. |
| Vnudge | one system-prompt line: "no analysis or prose, call submit_structured_output immediately" |

A "limited reasoning" variant does not exist for this model. Poolside's release notes say:
"Laguna S 2.1 supports two thinking settings: off and max … low, medium, and high effort
controls are not available". Thinking-off is already in effect on the forced call:
`reasoning_tokens` was 0 and `reasoning_content` was empty on every V0 and T0.7 call. So
`reasoning:{effort:…}` has nothing to reduce. `include_reasoning:false` only hides reasoning,
it does not stop the spend.

## Results

| variant | input | n | cut at 14k | HTTP 500 | out tokens mean (200s) | latency s mean | called the function | valid on replay |
|---|---|---|---|---|---|---|---|---|
| V0 | c80b | 4 | 2 | 0 | 8,929 | 148.7 | 2 | 2 |
| T0.7 | c80b | 4 | 0 | 0 | 3,212 | 61.1 | 4 | 4 |
| Vnudge | c80b | 4 | 0 | 0 | 3,058 | 55.6 | 4 | 4 |
| Von | c80b | 4 | 4 | 0 | 14,000 | 228.6 | 0 | 0 |
| V0 | 322b | 3 | 2 | 1 | 14,000 | 212.8 | 0 | 0 |
| T0.7 | 322b | 4 | 1 | 1 | 7,641 | 157.6 | 2 | 2 |
| V0 | ede4 | 2 | 0 | 0 | 986 | 22.6 | 2 | 2 |
| T0.7 | ede4 | 2 | 0 | 0 | 878 | 19.8 | 2 | 2 |

The HTTP 500s arrived after 206 s and 258 s. That matches the known ~275 s server-side failure
of a runaway generation, so they are counted as failures.

**What the loop is.** With thinking off, Laguna thinks out loud in `content`: "I'll analyze
this session… Let me… Actually, I want to reconsider…". It writes 59–69k characters with a
zlib ratio of 0.06–0.12 (heavy repetition) and never calls the function. The endpoint does not
enforce the forced `tool_choice`. With thinking on (Von), all 14,000 tokens go into
`reasoning_content` and `content` stays empty, 4 of 4 times.

## Readings, with their strength

- **Thinking on always loops (strong).** Von was cut 4 of 4 times, against 0 of 4 for T0.7 on
  the same input (Fisher p = 0.029). Production confirms it: at 21:23, 21:29 and 21:35Z,
  consolidate booked `http_calls=2` with 15.8–16.7k output tokens. That is a prose reply
  followed by this fallback cut at 14,000. The fallback did rescue one miss, at 21:14:41Z
  (`ok`, 9,798 output, `http_calls=2`), so the production record is 1 rescued and 3 cut. It was
  removed anyway: the retry that replaces it is a forced call at 0.7, which was cut 0 of 4
  times on c80b, against 4 of 4 for the thinking-on shape. A rescue also still cost up to 14k.
- **Temperature 0.7 reduces failures but does not cure them.** On the two looping inputs
  pooled, V0 failed 5 of 7 and T0.7 failed 2 of 8 (Fisher p = 0.13). On c80b alone it was
  2 of 4 against 0 of 4 (p = 0.43). The direction is consistent with Poolside's own recipe
  (0.7). The measurement is not significant at n ≤ 8.
- **Vnudge matched T0.7 on c80b (0 of 4)** but was not tested on 322b because the budget ran
  out. It is not shipped.
- **ede4 did not reproduce** in 4 of 4 runs.
- **Every finished answer, in every variant, passed ai-memory's validation on replay.**
- **Cut answers were checked for salvaged drafts.** Since c09b061, the `content` of a cut answer
  becomes the `Truncated` partial that auto_improve salvages from, and with thinking off that
  `content` is the model's deliberation. Three cut V0 bodies (c80b ×1, 322b ×2) were replayed on
  b50cd18b: each gave "0 proposals recovered". No draft became a proposal. n = 3; this is not a
  proof.

## What shipped in code (branch `fix/laguna-truncation-classify`)

1. **Per-provider temperature.** `POOLSIDE_TEMPERATURE = 0.7` is forced on every request
   (forced-tool and text) for `poolside/*` models. Other models keep the caller's value.
   Override with `with_temperature_override`.
2. **No text fallback for Poolside.** A forced-tool reply without a usable call becomes a shape
   error after one request. Other models keep the fallback. Opt back in with
   `with_tool_text_fallback(true)`.
3. **Breaker per input.** The lane's zero-yield breaker counts each input once per streak. The
   same session cut repeatedly no longer pauses the key for every caller. Quota and 5xx still
   count as before.
4. **Cut reported over decline.** When a free lane cut the request and the metered lane then
   declined it, the chain returns `Truncated` instead of `MeteredDeclined`. The SessionEnd
   worker then parks the generation after 2 attempts (`8b41b06`) instead of retrying 5 times.
   A decline with no cut behind it, such as after a quota wall, stays `MeteredDeclined`. This is
   pinned by an invariant: the park budget (2) is below the zero-yield threshold (3).
