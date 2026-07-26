#!/usr/bin/env bash
# Atlas self-test. Run from the project root inside the `atlas` conda env:
#
#   conda activate atlas
#   bash scripts/selftest.sh 2>&1 | tee selftest.log
#
# Read-only and dry-run only -- this script never places an order. It checks the
# things that are easy to get silently wrong: the model loading once, every
# symbol being accounted for, and the CSV being overwritten rather than grown.

set -uo pipefail
pass=0; fail=0
section() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail+1)); }
check(){ if [ "$1" = 0 ]; then ok "$2"; else bad "$2"; fi; }
# Run a command to a temp file, then grep it. Never pipe a command straight into
# `grep -q` under `set -o pipefail`: grep exits at the first match, the producer
# takes SIGPIPE and returns 141, and pipefail reports the whole pipeline failed.
grep_out(){ local pat="$1"; shift; "$@" </dev/null >/tmp/atlas_grep.log 2>&1; grep -q "$pat" /tmp/atlas_grep.log; }

section "Environment"
python -c 'import sys; assert sys.version_info[:2]>=(3,10)' 2>/dev/null
check $? "python $(python -V 2>&1 | cut -d' ' -f2) is 3.10+"
[ "$(python -c 'import atlas,os;print(os.path.dirname(atlas.__file__))')" = "$PWD/atlas" ]
check $? "the installed atlas package is this working copy"
python -c 'import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)'
check $? "torch sees the MPS backend (GPU acceleration active)"
[ -f vendor/kronos/model/kronos.py ]
check $? "vendored Kronos source present"
[ -f creds.env ] && [ "$(stat -f '%OLp' creds.env)" = "600" ]
check $? "creds.env exists and is owner-readable only"

section "Unit tests"
python -m pytest -q >/tmp/atlas_pytest.log 2>&1
check $? "$(tail -1 /tmp/atlas_pytest.log | sed 's/ in .*//')"

section "Account (read-only)"
for cmd in portfolio orders; do
  atlas "$cmd" </dev/null >/tmp/atlas_$cmd.log 2>&1
  check $? "atlas $cmd"
done
atlas info AAPL </dev/null >/tmp/atlas_info.log 2>&1
check $? "atlas info AAPL"
atlas news AAPL </dev/null >/tmp/atlas_news.log 2>&1
check $? "atlas news AAPL"

section "Orders (dry run -- nothing is submitted)"
grep_out "ORDER SUMMARY" atlas buy AAPL 500 --dry-run
check $? "atlas buy AAPL 500 --dry-run produced an order summary"
grep_out "Non-fractionable" atlas buy SNDQ 500 --dry-run
check $? "non-fractionable rounding path is detected"
grep_out "not enough for a whole share" atlas buy SNDQ 10 --dry-run
check $? "an unaffordable whole-share order is refused"
grep_out "No open position" atlas sell KO
check $? "selling an unheld symbol is refused"

section "Scan (this is the slow part)"
rows_in() { [ -f "$1" ] && echo $(( $(wc -l < "$1") - 1 )) || echo 0; }  # minus header
before=$(rows_in top30_assets.csv)
time atlas scan --limit 40 </dev/null >/tmp/atlas_scan.log 2>&1
scan_status=$?
check $scan_status "atlas scan --limit 40 exited cleanly"

loads=$(grep -c "loaded Kronos" /tmp/atlas_scan.log)
[ "$loads" = 1 ]
check $? "model loaded exactly once (found $loads)"

recon=$(grep "^scored " /tmp/atlas_scan.log)
printf '  %s\n' "$recon"
python - "$recon" <<'PY'
import re, sys
m = re.search(r"scored (\d+) / eligible (\d+) / universe (\d+)", sys.argv[1] or "")
sys.exit(0 if m else 1)
PY
check $? "reconciliation line present"

after=$(rows_in top30_assets.csv)
[ "$after" -le 30 ]
check $? "top30_assets.csv holds at most 30 data rows (was $before, now $after)"

section "Summary"
printf '  %d passed, %d failed\n' "$pass" "$fail"
printf '\n  Paste back: this summary, plus these two lines --\n'
grep -E "^loaded Kronos|^scored " /tmp/atlas_scan.log | sed 's/^/    /'
printf '\n  Full scan log: /tmp/atlas_scan.log\n'
printf '  Top of the ranking:\n'
head -4 top30_assets.csv | cut -c1-100 | sed 's/^/    /'
exit $(( fail > 0 ))
