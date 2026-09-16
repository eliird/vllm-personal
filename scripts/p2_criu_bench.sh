#!/usr/bin/env bash
# Phase 2 CRIU memory sweep. For each size: start the probe, read baseline,
# criu dump, kill (implicit), criu restore, re-read, record times + image size.
#
# Env: SIZES="1024 8192 16384"   (MiB)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SIZES="${SIZES:-1024 8192 16384}"
[ -x scripts/p2_ram ] || gcc -O2 scripts/p2_ram.c -o scripts/p2_ram

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

for MB in $SIZES; do
  D="/tmp/p2_$MB"; rm -rf "$D"; mkdir -p "$D/images"
  scripts/p2_ram "$MB" "$D/state" > "$D/probe.log" 2>&1 &
  PID=$!
  for _ in $(seq 100); do [ -s "$D/state" ] && break; sleep 0.2; done
  sleep 3
  rss=$(awk '/VmRSS/{print $2}' "/proc/$PID/status" 2>/dev/null)
  before=$(cat "$D/state")

  t0=$(date +%s%N)
  $SUDO criu dump --shell-job --images-dir "$D/images" --tree "$PID" 2>&1 | tee "$D/dump.log"
  du=$?
  t1=$(date +%s%N)
  $SUDO criu restore --shell-job --restore-detached --images-dir "$D/images" 2>&1 | tee "$D/restore.log"
  t2=$(date +%s%N)

  sleep 3
  after=$(cat "$D/state")
  img_kb=$(du -sk "$D/images" | cut -f1)
  dump_ms=$(( (t1 - t0) / 1000000 ))
  restore_ms=$(( (t2 - t1) / 1000000 ))

  b_ctr=${before%% *}; b_sum=${before##* }
  a_ctr=${after%% *};  a_sum=${after##* }
  if [ "${a_ctr:-0}" -gt "${b_ctr:-0}" ] 2>/dev/null && [ "$a_sum" = "$b_sum" ]; then
    verdict=PASS
  else
    verdict=FAIL
  fi
  printf 'MB=%s RSS_kb=%s dump_ms=%s restore_ms=%s image_kb=%s before=[%s] after=[%s] %s\n' \
    "$MB" "${rss:-?}" "$dump_ms" "$restore_ms" "$img_kb" "$before" "$after" "$verdict"
  kill "$PID" 2>/dev/null || true
done
