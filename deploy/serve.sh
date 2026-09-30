#!/bin/sh
# fn8 wrapper for the ai-memory server (com.fn8.ai-memory LaunchAgent).
#
# ref: scripts/bin/ai-memory-serve.sh:27-51 (pre-rewrite, the `case "$PICK"` block) ·
#      scripts/bin/ai-memory-serve.sh:44-49 (pre-rewrite, the duplicated embeddings+exec) ·
#      scripts/bin/ai-memory-router.sh (route_pick, _route_key) ·
#      scripts/bin/fn8-ai-memory-provider-gate.sh (provider_gate_decide) ·
#      scripts/lib/fn8-lane-wall.mjs + scripts/bin/fn8-lane-wall.mjs (passive health)
#
# ONE SELECTOR, ONE BINDING, ONE EXEC. Lanes are tried in the operator-ratified order
# POOLSIDE (key A, then key B) -> GEMINI -> OPENCODE ZEN -> no-LLM (poolside first since 2026-09-23: free vs metered), each admitted by the SAME three-verdict provider
# gate, and exactly one of them is bound before a single exec at the bottom.
#
# WHAT THIS REPLACED, and why the shape matters. The previous version tried opencode FIRST and
# treated gemini as a last resort, and it carried a structural defect: the `zen:*` branch at
# :27-51 held a second embeddings block and a second exec at :44-49, so that branch short-circuited
# every policy below it — including the gemini gate — and ran even when its key lookup returned
# empty. Any future code added below would have been silently bypassed too. Found by an adversarial
# design pass, not by the file failing.
#
# Mental model, unchanged and still right: route around failure, never wait (Stefan 2026-08-24).
# What changed is WHICH failure. opencode read as a dead vendor for nine days — 18 of 18 probes
# returning 400 — and it was namespace drift in nine stale model names, not an outage. The router
# now resolves its roster from the live free-model cache, so "opencode last" is a priority decision,
# not a verdict on the vendor.
#
# Secrets never land in plist, repo, or argv.
set -eu

export AI_MEMORY_DATA_DIR="$HOME/.fn8/ai-memory"
. "$HOME/.fn8/ai-memory/router.sh"

FN8_REPO="${FN8_REPO_ROOT:-/Users/stefantalos/My Space/Fn8 - Projects/fn8-os}"
PROVIDER_GATE="$HOME/.fn8/ai-memory/fn8-ai-memory-provider-gate.sh"
[ -r "$PROVIDER_GATE" ] || PROVIDER_GATE="$FN8_REPO/scripts/bin/fn8-ai-memory-provider-gate.sh"
WALL_CLI="$FN8_REPO/scripts/bin/fn8-lane-wall.mjs"
NODE_BIN="${FN8_NODE_BIN:-/usr/local/bin/node}"
BUS="${FN8_CHYROS_ACTIONABLE_PATH:-/tmp/fn8-chyros-actionable.jsonl}"

LANE=""; LANE_PROVIDER=""; LANE_MODEL=""; LANE_KEY=""; LANE_BASE=""; SKIPPED=""

note() { echo "[serve.sh] $*" >&2; }

# A gate we cannot read is UNKNOWN health, never permission to proceed. Fail closed, and say so.
if [ -r "$PROVIDER_GATE" ]; then
    # shellcheck source=/dev/null
    . "$PROVIDER_GATE"
    GATE_OK=1
else
    note "provider gate missing at $PROVIDER_GATE — every lane will be withheld"
    GATE_OK=0
fi

# The passive wall vetoes a lane REAL traffic has convicted (exit 1 = ejected). Anything else,
# including a lane with no evidence at all, passes through to the active gate: "never judged" is not
# "failed", and vetoing on thin evidence would deadlock a fresh install. The wall never blocks a
# boot — a missing node or an unwritable ledger simply means no veto.
wall_allows() {
    [ -x "$NODE_BIN" ] && [ -f "$WALL_CLI" ] || return 0
    "$NODE_BIN" "$WALL_CLI" verdict --lane "$1" >/dev/null 2>&1 || return 1
    return 0
}

# quota_wall_until <key_file> — the Unix second until which the ai-memory binary's quota book
# (FN8-8120, llm-quota-state.json, written by the binary on a `429 usage limit exceeded`) walls this
# key, looked up by key fingerprint (sha256 of the trimmed key, 8 hex — the ledger's key_fp). Prints
# nothing when the key is not walled, the file is absent/unreadable, or the wall has expired: no
# state is never "blocked". This runs BEFORE provider_gate_decide so a key known to be walled until
# its daily reset costs no 24k-token volume probe at boot. Key value never printed.
QUOTA_STATE="${AI_MEMORY_LLM_QUOTA_STATE_PATH:-${AI_MEMORY_DATA_DIR:-$HOME/.fn8/ai-memory}/llm-quota-state.json}"
quota_wall_until() {
    [ -r "$QUOTA_STATE" ] && [ -r "$1" ] || return 0
    python3 -c '
import hashlib, json, sys
try:
    fp = hashlib.sha256(open(sys.argv[1]).read().strip().encode()).hexdigest()[:8]
    q = (json.load(open(sys.argv[2])).get("keys") or {}).get(fp) or {}
    u = q.get("blocked_until")
    if isinstance(u, int) and u > int(sys.argv[3]):
        print(u)
except Exception:
    pass
' "$1" "$QUOTA_STATE" "$(date -u +%s)" 2>/dev/null || true
}

# try_lane <lane> <provider> <model> <key_file> [base_url]
# Every failure records a reason and returns 1 so selection CONTINUES — under `set -eu` an
# unhandled non-zero here would kill the wrapper before the fallbacks ever ran.
try_lane() {
    _l="$1"; _p="$2"; _m="$3"; _k="$4"; _b="${5:-}"
    if ! wall_allows "$_l"; then SKIPPED="$SKIPPED $_l(ejected-by-traffic)"; return 1; fi
    if [ "$GATE_OK" -ne 1 ]; then SKIPPED="$SKIPPED $_l(no-gate)"; return 1; fi
    if [ ! -r "$_k" ] || [ ! -s "$_k" ]; then SKIPPED="$SKIPPED $_l(no-key)"; return 1; fi
    _qu=$(quota_wall_until "$_k")
    if [ -n "$_qu" ]; then
        SKIPPED="$SKIPPED $_l(quota-until:$(date -u -r "$_qu" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "$_qu"))"
        return 1
    fi
    _d=$(provider_gate_decide "$HOME/.fn8/ai-memory/${_l}-breaker.json" "$_k" "$_m" \
                              "$(date -u +%s)" "$_p" "$_b" 2>/dev/null || echo withhold)
    if [ "$_d" = "export" ]; then
        LANE="$_l"; LANE_PROVIDER="$_p"; LANE_MODEL="$_m"; LANE_KEY="$_k"; LANE_BASE="$_b"
        return 0
    fi
    SKIPPED="$SKIPPED $_l(gate:$_d)"; return 1
}

# lane_admissible <lane> <key_file> — a NON-BINDING admissibility check for the runtime fallback
# lane. It never touches LANE*, and it makes no network call: the provider gate's volume probe
# would spend a metered Gemini request on every boot just to arm a lane that may never be used.
# Admits when the wall has not ejected the lane, the gate is loaded, the key file is present, and
# the lane's breaker is not open. The binary's own circuit judges the lane on real traffic.
lane_admissible() {
    wall_allows "$1" || return 1
    [ "$GATE_OK" -eq 1 ] || return 1
    [ -r "$2" ] && [ -s "$2" ] || return 1
    [ "$(breaker_gate "$HOME/.fn8/ai-memory/${1}-breaker.json" "$(date -u +%s)" 2>/dev/null || echo deny)" = allow ]
}

# ── SELECTION, in the ratified order ─────────────────────────────────────────────────────────────
GEMINI_KEY="$HOME/.fn8-secrets/gemini-api-key"
POOLSIDE_KEY="${FN8_POOLSIDE_KEY_FILE:-$HOME/.fn8-secrets/poolside-api-key}"
# Second Poolside account (operator decision 2026-09-24: use BOTH Laguna keys). Sunk from
# ~/.config/poolside/credentials.json by fingerprint (153b1d4d), mode 0600; never read here.
POOLSIDE_KEY_2="${FN8_POOLSIDE_KEY_FILE_2:-$HOME/.fn8-secrets/poolside-api-key-2}"
POOLSIDE_BASE="${FN8_POOLSIDE_BASE_URL:-https://inference.poolside.ai/v1}"

# Poolside first, Gemini second (operator ruling 2026-09-23: Laguna is free, the Gemini API is
# metered prepay). Laguna structured output works since ai-memory e0251d5 + 6faab6b (forced tool
# call; 6/6 live auto_improve runs validated), so the free lane no longer costs quality.
# Key B is its own lane (`poolside-b`, own breaker file): the boot gate probes with the key it
# is given, so a key-A quota wall must not keep the free lane from binding on key B.
try_lane poolside openai-compat "${FN8_POOLSIDE_MODEL:-poolside/laguna-s-2.1}" "$POOLSIDE_KEY" \
       "$POOLSIDE_BASE" \
  || try_lane poolside-b openai-compat "${FN8_POOLSIDE_MODEL:-poolside/laguna-s-2.1}" "$POOLSIDE_KEY_2" \
       "$POOLSIDE_BASE" \
  || try_lane gemini gemini "${FN8_AI_MEMORY_GEMINI_MODEL:-gemini-2.5-flash}" "$GEMINI_KEY" \
  || true

# OpenCode last. Its own router picks account+model (now from the live free roster), and
# router-force still overrides everything when a batch job has pinned a lane deliberately.
if [ -z "$LANE" ]; then
    FORCE_FILE="$HOME/.fn8/ai-memory/router-force"
    if [ -s "$FORCE_FILE" ]; then
        PICK=$(cat "$FORCE_FILE"); note "router-force active: $PICK"
    else
        PICK=$(route_pick || true)
    fi
    if [ -n "${PICK:-}" ]; then
        case "$PICK" in
            zen:*)
                Z_ACCT=${PICK%% *}; Z_ACCT=${Z_ACCT#zen:}; Z_MODEL=${PICK#* }
                _zk=$(_route_key "$Z_ACCT" || true)
                if [ -n "${_zk:-}" ]; then
                    LLM_API_KEY="$_zk"; export LLM_API_KEY
                    export LLM_BASE_URL="https://opencode.ai/zen/v1"
                    LANE="zen:$Z_ACCT"; LANE_PROVIDER=openai-compat; LANE_MODEL="$Z_MODEL"
                else
                    SKIPPED="$SKIPPED zen(no-key)"
                fi ;;
            *)
                R_ACCT=${PICK%% *}; R_MODEL=${PICK#* }
                _ok=$(_route_key "$R_ACCT" || true)
                if [ -n "${_ok:-}" ]; then
                    OPENCODE_API_KEY="$_ok"; export OPENCODE_API_KEY
                    LANE="$R_ACCT"; LANE_PROVIDER=opencode; LANE_MODEL="$R_MODEL"
                else
                    SKIPPED="$SKIPPED opencode(no-key)"
                fi ;;
        esac
    else
        SKIPPED="$SKIPPED opencode(no-candidate)"
    fi
fi

# ── BINDING, exactly once ────────────────────────────────────────────────────────────────────────
if [ -n "$LANE" ]; then
    case "$LANE_PROVIDER" in
        gemini)        GEMINI_API_KEY=$(cat "$LANE_KEY"); export GEMINI_API_KEY ;;
        openai-compat)
            if [ -n "$LANE_KEY" ] && [ -r "$LANE_KEY" ]; then
                LLM_API_KEY=$(cat "$LANE_KEY"); export LLM_API_KEY
                export LLM_BASE_URL="$LANE_BASE"
            fi ;;
        opencode)      : ;;   # OPENCODE_API_KEY already exported above
    esac
    export AI_MEMORY_LLM_PROVIDER="$LANE_PROVIDER"
    export AI_MEMORY_LLM_MODEL="$LANE_MODEL"
    export AI_MEMORY_ROUTER_ACCOUNT="$LANE"
    export AI_MEMORY_CONSOLIDATE_ON_SESSION_END=true
    # RUNTIME FALLBACK (FN8-8120). The lane above is bound once per boot; before this, a Poolside
    # 429 ("usage limit exceeded", measured 2026-09-24 00:11Z) or 500 stalled auto_improve until a
    # restart even though Gemini is the approved second lane. When the bound lane is poolside and
    # gemini is admissible, hand Gemini to the binary as its runtime fallback: it retries a
    # 429/5xx/timeout/truncated request on Gemini, skips Poolside for 15 min after a 429 (or 3
    # consecutive 5xx), then probes it again. The key travels as a FILE PATH, never a value.
    # Binaries older than ai-memory 2f03ead ignore these variables, so either deploy order is safe.
    FALLBACK_NOTE=""
    # SECOND POOLSIDE KEY as a runtime lane (binary: AI_MEMORY_LLM_EXTRA_API_KEY_FILES). The binary
    # tries the bound key, then the other key, then Gemini; a 429 pauses only that key for 15 min.
    # Admitted on presence + wall only: it is the same free vendor, and the binary's per-key breaker
    # judges it on real traffic. Path only; the value never passes through this script.
    unset AI_MEMORY_LLM_EXTRA_API_KEY_FILES 2>/dev/null || true
    case "$LANE" in
        poolside)   _other_lane=poolside-b; _other_key="$POOLSIDE_KEY_2" ;;
        poolside-b) _other_lane=poolside;   _other_key="$POOLSIDE_KEY" ;;
        *)          _other_lane=""; _other_key="" ;;
    esac
    if [ -n "$_other_key" ] && [ -r "$_other_key" ] && [ -s "$_other_key" ] && wall_allows "$_other_lane"; then
        export AI_MEMORY_LLM_EXTRA_API_KEY_FILES="$_other_key"
        FALLBACK_NOTE=" extra_key=$_other_lane"
    fi
    if { [ "$LANE" = poolside ] || [ "$LANE" = poolside-b ]; } && lane_admissible gemini "$GEMINI_KEY"; then
        export AI_MEMORY_LLM_FALLBACK_PROVIDER=gemini
        export AI_MEMORY_LLM_FALLBACK_MODEL="${FN8_AI_MEMORY_GEMINI_MODEL:-gemini-2.5-flash}"
        export AI_MEMORY_LLM_FALLBACK_API_KEY_FILE="$GEMINI_KEY"
        FALLBACK_NOTE="$FALLBACK_NOTE fallback=gemini/$AI_MEMORY_LLM_FALLBACK_MODEL"
    else
        unset AI_MEMORY_LLM_FALLBACK_PROVIDER AI_MEMORY_LLM_FALLBACK_MODEL \
              AI_MEMORY_LLM_FALLBACK_BASE_URL AI_MEMORY_LLM_FALLBACK_API_KEY_FILE 2>/dev/null || true
    fi
    # ZERO INVISIBLE SPEND (floo charter §1). The binary appends every LLM call to
    # $AI_MEMORY_DATA_DIR/llm-calls.jsonl (lane, key fingerprint, caller, tokens, estimate), copies
    # metered (Gemini) calls into floo's harness-presence ledger, and posts breaker alarms to the bus.
    # Entity: the Gemini API key is the stefan.talos@gmail.com project per the floo roster §2.1.
    FLOO_PRESENCE="${FN8_FLOO_PRESENCE_PATH:-/Users/stefantalos/My Space/Fn8 - Projects/floo/state/finops/harness-presence.jsonl}"
    if [ -d "$(dirname "$FLOO_PRESENCE")" ]; then
        export AI_MEMORY_FINOPS_PRESENCE_PATH="$FLOO_PRESENCE"
        export AI_MEMORY_METERED_ENTITY="${FN8_AI_MEMORY_METERED_ENTITY:-Personal}"
    fi
    export AI_MEMORY_LLM_ALARM_PATH="$BUS"
    # JEV VALUE GATE (operator decision 2026-09-24): before an auto_improve/experience request
    # reaches the METERED lane (Gemini), Jev answers atomic value questions at the 0.9 money bar;
    # a no, an emulated answer or an unavailable judge keeps it off Gemini (fail closed) and the
    # session waits for free Laguna. Availability routing stays in code. The Jev CLI books its
    # own OpenRouter spend via floo's authoriseMeteredCall.
    JEV_CLI="${FN8_FLOO_JEV_CLI:-/Users/stefantalos/My Space/Fn8 - Projects/floo/scripts/bin/floo-jev.mjs}"
    if [ -f "$JEV_CLI" ] && [ -x "$NODE_BIN" ]; then
        export AI_MEMORY_METERED_GATE_JEV_SCRIPT="$JEV_CLI"
        export AI_MEMORY_METERED_GATE_NODE="$NODE_BIN"
        export AI_MEMORY_METERED_GATE_CALLERS=auto_improve,experience,consolidate
    else
        unset AI_MEMORY_METERED_GATE_JEV_SCRIPT AI_MEMORY_METERED_GATE_NODE 2>/dev/null || true
    fi
    note "lane=$LANE provider=$LANE_PROVIDER model=$LANE_MODEL$FALLBACK_NOTE${SKIPPED:+ (skipped:$SKIPPED)}"
else
    # NO LANE. ai-memory falls back to a labelled `_Synthesised by ai-memory (M3, no-LLM
    # heuristic)._` page, which is the CORRECT outcome: an honest mechanical summary beats a
    # confabulated one written into a store later read as fact. No panic promotion.
    #
    # Inherited provider variables are cleared ONLY here. Doing it unconditionally would wipe the
    # opencode/zen keys exported during selection a few lines above — the bug this placement avoids.
    unset GEMINI_API_KEY LLM_API_KEY LLM_BASE_URL AI_MEMORY_LLM_PROVIDER AI_MEMORY_LLM_MODEL \
          AI_MEMORY_LLM_FALLBACK_PROVIDER AI_MEMORY_LLM_FALLBACK_MODEL \
          AI_MEMORY_LLM_FALLBACK_BASE_URL AI_MEMORY_LLM_FALLBACK_API_KEY_FILE \
          AI_MEMORY_LLM_EXTRA_API_KEY_FILES AI_MEMORY_METERED_GATE_JEV_SCRIPT \
          AI_MEMORY_METERED_GATE_NODE 2>/dev/null || true
    note "NO LLM lane bound${SKIPPED:+ — skipped:$SKIPPED}"
    printf '{"ts":"%s","actionable":true,"source":"ai-memory-serve","severity":"MED","reason":"llm-provider-withheld","recommended_action":"ai-memory started with NO LLM provider. Lanes tried in order poolside (key A), poolside-b (key B), gemini, opencode zen -- none passed. Skipped:%s. auto_improve will not generate proposals; consolidation falls back to the labelled heuristic page. Check the per-lane breaker files in ~/.fn8/ai-memory/."}\n' \
        "$(date -u +%FT%TZ)" "${SKIPPED:- none}" >> "$BUS" 2>/dev/null || true
fi

# The binding is recorded whether or not a lane was found: lane=none is a first-class fact, and a
# wall that only hears about successes cannot measure anything. Never allowed to fail a boot.
if [ -x "$NODE_BIN" ] && [ -f "$WALL_CLI" ]; then
    "$NODE_BIN" "$WALL_CLI" epoch --lane "${LANE:-none}" \
        ${LANE_PROVIDER:+--provider "$LANE_PROVIDER"} ${LANE_MODEL:+--model "$LANE_MODEL"} \
        --reason "${SKIPPED:-first-choice}" >/dev/null 2>&1 || true
fi

# Local ONNX embeddings (Step 4, D-ai-memory §RAG lesson): curated-page layer only, served by
# ~/.fn8/ai-memory/embed-server.mjs (transformers.js, multilingual-e5-small 384d).
export AI_MEMORY_EMBEDDING_PROVIDER=openai-compat
export AI_MEMORY_EMBEDDING_BASE_URL="http://127.0.0.1:49375/v1"
export AI_MEMORY_EMBEDDING_MODEL="multilingual-e5-small"
export AI_MEMORY_EMBEDDING_DIM=384

exec "$HOME/Applications/ai-memory/ai-memory" serve --transport http --bind 127.0.0.1:49374
