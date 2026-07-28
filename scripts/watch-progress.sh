#!/usr/bin/env bash
# Live progress bar for a long Atlas run started in another terminal.
#
#   bash scripts/watch-progress.sh
#
# Long commands write data/progress.<pid>.json once a second. Rich suppresses
# its own bar when stdout is redirected, so a backgrounded job is otherwise
# indistinguishable from a hung one.
#
# The pid is in the filename so two concurrent runs cannot overwrite each
# other's counts. This watcher tracks the most recently updated one, and
# reports a dead process rather than showing its last count forever.
#
# Safe to start, stop and restart at any time -- it only ever reads.

set -uo pipefail
DIR="${1:-data}"

# The bar width is derived from the terminal, not fixed. A line longer than the
# terminal wraps, and then `\r` only returns to the start of the last visual
# row -- so every refresh prints a fresh bar instead of overwriting the old one.
term_width() { local c; c=$(tput cols 2>/dev/null || echo 80); echo "$c"; }
CLEAR=$'\033[2K'   # clear the whole line, so a shorter frame cannot leave debris

fmt() {  # seconds -> 1h02m / 5m03s / 42s
  local s=${1:-0}
  if   [ "$s" -ge 3600 ]; then printf '%dh%02dm' $((s/3600)) $(((s%3600)/60))
  elif [ "$s" -ge 60   ]; then printf '%dm%02ds' $((s/60)) $((s%60))
  else printf '%ds' "$s"; fi
}

printf 'watching %s/progress.*.json -- ctrl-c to stop\n\n' "$DIR"
seen=0
while true; do
  # Prefer a heartbeat whose process is actually alive. Picking merely the
  # newest file means a just-finished short run can mask a long one still going
  # -- which is exactly how this watcher once announced "done" mid-run.
  FILE=""
  for f in $(ls -t "$DIR"/progress.*.json 2>/dev/null); do
    fpid=$(basename "$f" | sed 's/progress\.\([0-9]*\)\.json/\1/')
    if [ -n "$fpid" ] && kill -0 "$fpid" 2>/dev/null; then FILE="$f"; break; fi
  done
  # Nothing live: fall back to the newest so a finished run still reports.
  [ -z "$FILE" ] && FILE=$(ls -t "$DIR"/progress.*.json 2>/dev/null | head -1)

  if [ -z "$FILE" ]; then
    if [ "$seen" = 1 ]; then printf '\n\ndone -- the run finished.\n'; exit 0; fi
    printf '\rno run in progress (waiting)…                                    '
    sleep 2; continue
  fi

  # Tab-delimited, not whitespace: an intraday `last_date` is "2026-02-04 15:35"
  # and the space inside it shifts every later field, so `pid` ends up holding
  # fragments of the timestamp and the liveness check fails on a healthy run.
  IFS=$'\t' read -r done total task last elapsed eta pid < <(
    python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print("\t".join(str(x) for x in (
    d.get("done",0), d.get("total",1), d.get("task","run"),
    d.get("last_date","-"), int(d.get("elapsed_s") or 0), int(d.get("eta_s") or 0),
    d.get("pid",0))))
' "$FILE" 2>/dev/null) || { sleep 2; continue; }
  seen=1

  # A heartbeat whose process is gone means the run was killed, not finished --
  # showing its final count indefinitely would be misleading.
  if [ "$pid" != 0 ] && ! kill -0 "$pid" 2>/dev/null; then
    printf '\n\nprocess %s is gone -- the run was interrupted at %s/%s.\n' "$pid" "$done" "$total"
    printf 'stale heartbeat: %s (safe to delete)\n' "$FILE"
    exit 1
  fi

  pct=$(( done * 100 / (total > 0 ? total : 1) ))

  # Build the text first, then size the bar to whatever room is left. Recomputed
  # every frame so resizing the window mid-run stays correct.
  suffix=$(printf '%3d%%  %s/%s  %s elapsed  eta %s  @%s' \
    "$pct" "$done" "$total" "$(fmt "$elapsed")" "$(fmt "$eta")" "$last")
  prefix="$task"
  cols=$(term_width)
  # prefix + " [" + bar + "] " + suffix, and one spare column so a full line
  # never triggers the terminal's own wrap.
  width=$(( cols - ${#prefix} - ${#suffix} - 5 ))
  [ "$width" -lt 8 ] && width=8
  [ "$width" -gt 60 ] && width=60

  filled=$(( done * width / (total > 0 ? total : 1) ))
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $((width-filled)) '' | tr ' ' '.')

  printf '\r%s%s [%s] %s' "$CLEAR" "$prefix" "$bar" "$suffix"

  [ "$done" -ge "$total" ] && { printf '\n\ndone.\n'; exit 0; }
  sleep 2
done
