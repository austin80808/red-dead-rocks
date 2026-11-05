#!/bin/bash
# Run Unreal for a Red Rocks project copy while holding the shared Unreal lock.
#
#   tools/ue.sh build     <project-dir>                 Build.sh RedRocksEditor
#   tools/ue.sh cpp-tests <project-dir> [filter]        headless automation tests (default filter "RedRocks.")
#   tools/ue.sh playtest  <project-dir> [SCENARIOS=a,b] Play In Editor: unreal/playtest_explore.py
#   tools/ue.sh rebuild   <project-dir>                 unreal/rebuild_level.py (wipes that copy's level)
#   tools/ue.sh py        <project-dir> <script.py>     any editor Python script, headless
#
# Every subcommand waits for ~/.local/bin/unreal-lock (Hanging Lake shares this
# Unreal install) and releases it on exit. Logs go to /tmp/redrocks-<copy>/ so
# runs in different copies never overwrite each other's logs.
set -uo pipefail

UE_ROOT="/Users/Shared/Epic Games/UE_5.8/Engine"
EDITOR="$UE_ROOT/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
EDITOR_CMD="$UE_ROOT/Binaries/Mac/UnrealEditor-Cmd"
LOCK="$HOME/.local/bin/unreal-lock"
OWNER="${UE_LOCK_OWNER:-Red Rocks controllers}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

usage() { sed -n 2,13p "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ $# -ge 2 ] || usage
sub="$1"
proj_dir="$(cd "$2" 2>/dev/null && pwd)" || { echo "no such project dir: $2" >&2; exit 2; }
shift 2
uproject="$proj_dir/RedRocks.uproject"
[ -f "$uproject" ] || { echo "no RedRocks.uproject in $proj_dir" >&2; exit 2; }

copy="$(basename "$proj_dir")"
export REDROCKS_LOG_DIR="${REDROCKS_LOG_DIR:-/tmp/redrocks-$copy}"
export REDROCKS_UNREAL_DIR="$REPO/unreal"
if [ -z "${REDROCKS_OUT_DIR:-}" ]; then
  if [ -f "$REPO/out/manifest.json" ]; then export REDROCKS_OUT_DIR="$REPO/out"
  else export REDROCKS_OUT_DIR="/Users/austin/dev/redrocks-ue/out"; fi
fi
mkdir -p "$REDROCKS_LOG_DIR"

take_lock() {
  echo "waiting for the Unreal lock as \"$OWNER\"..."
  "$LOCK" wait "$OWNER" "$sub $copy" || { echo "could not take the Unreal lock" >&2; exit 1; }
  trap '"$LOCK" release "$OWNER" >/dev/null' EXIT
}

case "$sub" in
  build)
    take_lock
    "$UE_ROOT/Build/BatchFiles/Mac/Build.sh" RedRocksEditor Mac Development -Project="$uproject" -WaitMutex \
      > "$REDROCKS_LOG_DIR/build.log" 2>&1
    rc=$?
    grep -E "error:|Result:" "$REDROCKS_LOG_DIR/build.log" | tail -20
    exit $rc
    ;;
  cpp-tests)
    filter="${1:-RedRocks.}"
    report="$REDROCKS_LOG_DIR/cpp_tests"
    rm -rf "$report"
    take_lock
    "$EDITOR_CMD" "$uproject" -ExecCmds="Automation RunTests $filter;Quit" -unattended -nopause -nosplash \
      -nullrhi -NoSound -ReportExportPath="$report" > "$REDROCKS_LOG_DIR/cpp_tests.log" 2>&1
    python3 - "$report/index.json" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8-sig") as fh:
        data = json.load(fh)
except OSError:
    print("no test report written (did the editor start? see cpp_tests.log)")
    sys.exit(1)
tests = data.get("tests", [])
failed = [t for t in tests if t.get("state") != "Success"]
for t in failed:
    msgs = [e["event"]["message"] for e in t.get("entries", []) if e.get("event", {}).get("type") == "Error"]
    print("FAIL %s: %s" % (t.get("fullTestPath"), " | ".join(msgs) or t.get("state")))
print("%d passed, %d failed" % (len(tests) - len(failed), len(failed)))
sys.exit(1 if failed or not tests else 0)
PY
    ;;
  playtest)
    for kv in "$@"; do export "$kv"; done
    take_lock
    log="$REDROCKS_LOG_DIR/redrocks_playtest_explore.log"
    rm -f "$log"
    "$EDITOR" "$uproject" -nosplash -ExecCmds="py $REPO/unreal/playtest_explore.py" \
      > "$REDROCKS_LOG_DIR/playtest_editor.log" 2>&1
    [ -f "$log" ] || { echo "play-test wrote no log; see $REDROCKS_LOG_DIR/playtest_editor.log" >&2; exit 1; }
    cat "$log"
    ! grep -qE "^(FAIL|ERROR)" "$log"
    ;;
  rebuild)
    take_lock
    "$EDITOR_CMD" "$uproject" -run=pythonscript -script="$REPO/unreal/rebuild_level.py" -unattended -nosplash \
      -stdout > "$REDROCKS_LOG_DIR/rebuild_editor.log" 2>&1
    cat "$REDROCKS_LOG_DIR/redrocks_rebuild.log" 2>/dev/null || tail -20 "$REDROCKS_LOG_DIR/rebuild_editor.log"
    ! grep -q "FAIL" "$REDROCKS_LOG_DIR/redrocks_rebuild.log" 2>/dev/null
    ;;
  py)
    script="${1:?script path}"
    take_lock
    "$EDITOR_CMD" "$uproject" -run=pythonscript -script="$script" -unattended -nosplash -stdout \
      > "$REDROCKS_LOG_DIR/py_editor.log" 2>&1
    rc=$?
    grep -E "LogPython" "$REDROCKS_LOG_DIR/py_editor.log" | sed 's/^.*LogPython: //' | tail -80
    exit $rc
    ;;
  *) usage ;;
esac
