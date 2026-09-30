#!/bin/sh
# Harness: source quota_wall_until + try_lane from the patched serve.sh with stubs; a stub `curl`
# on PATH counts invocations. Walled key => zero curl calls and a quota-until skip reason.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"; printf '#!/bin/sh\necho x >> "%s/curl.count"\necho 200\n' "$T" > "$T/bin/curl"; chmod +x "$T/bin/curl"
printf 'sentinel-key-A\n' > "$T/keyA"; printf 'sentinel-key-B\n' > "$T/keyB"
FPA=$(python3 -c "import hashlib;print(hashlib.sha256(b'sentinel-key-A').hexdigest()[:8])")
NOW=$(date -u +%s)
export AI_MEMORY_DATA_DIR="$T"
# functions only: from 'quota_wall_until' block through end of try_lane
awk '/^QUOTA_STATE=/{on=1} on{print} on&&/^try_lane\(\)/{t=1} t&&/^}/{exit}' "$HERE/serve.sh" > "$T/fns.sh"
run() {
  ( PATH="$T/bin:$PATH"; SKIPPED=""; GATE_OK=1; LANE=""
    wall_allows() { return 0; }
    provider_gate_decide() { curl >/dev/null; echo export; }
    . "$T/fns.sh"
    if try_lane "$1" openai-compat m "$2" https://x; then echo "bound=$LANE"; else echo "skipped=$SKIPPED"; fi )
}
fail=0
# 1) walled key A: skipped, no curl
printf '{"version":1,"keys":{"%s":{"blocked_until":%s}}}' "$FPA" $((NOW+3600)) > "$T/llm-quota-state.json"
out=$(run poolside "$T/keyA"); echo "walled: $out"
case "$out" in *"poolside(quota-until:"*) ;; *) echo FAIL1; fail=1;; esac
[ ! -f "$T/curl.count" ] || { echo "FAIL1b curl called"; fail=1; }
# 2) the other key is not walled: bound, gate ran
out=$(run poolside-b "$T/keyB"); echo "other: $out"
[ "$out" = "bound=poolside-b" ] || { echo FAIL2; fail=1; }
# 3) expired wall: bound
printf '{"version":1,"keys":{"%s":{"blocked_until":%s}}}' "$FPA" $((NOW-5)) > "$T/llm-quota-state.json"
out=$(run poolside "$T/keyA"); echo "expired: $out"; [ "$out" = "bound=poolside" ] || { echo FAIL3; fail=1; }
# 4) corrupt state: bound (no state is never blocked)
echo '{ nope' > "$T/llm-quota-state.json"
out=$(run poolside "$T/keyA"); echo "corrupt: $out"; [ "$out" = "bound=poolside" ] || { echo FAIL4; fail=1; }
# 5) absent state: bound
rm -f "$T/llm-quota-state.json"
out=$(run poolside "$T/keyA"); echo "absent: $out"; [ "$out" = "bound=poolside" ] || { echo FAIL5; fail=1; }
[ $fail -eq 0 ] && echo ALL_PASS || { echo SOME_FAIL; exit 1; }
