#!/usr/bin/env bash
set -euo pipefail
PREFIX="${PREFIX:-/tmp/p0_plain}"
rm -rf "$PREFIX"
mkdir -p "$PREFIX"
cat > "$PREFIX/counter.sh" <<'EOF'
#!/usr/bin/env bash
n=0
while true; do n=$((n+1)); echo "$n" > "${PREFIX:-/tmp/p0_plain}/state"; sleep 1; done
EOF
chmod +x "$PREFIX/counter.sh"
PREFIX="$PREFIX" "$PREFIX/counter.sh" > "$PREFIX/counter.log" 2>&1 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; }
trap cleanup EXIT
sleep 3
before=$(cat "$PREFIX/state")
mkdir -p "$PREFIX/images"
dump_start=$(date +%s%N)
sudo criu dump --shell-job --images-dir "$PREFIX/images" --tree "$PID"
dump_end=$(date +%s%N)
sleep 2
restore_start=$(date +%s%N)
sudo criu restore --shell-job --restore-detached --images-dir "$PREFIX/images"
restore_end=$(date +%s%N)
sleep 3
after=$(cat "$PREFIX/state")
dump_ms=$(( (dump_end - dump_start) / 1000000 ))
restore_ms=$(( (restore_end - restore_start) / 1000000 ))
echo "dump_ms=$dump_ms restore_ms=$restore_ms"
echo "before=$before after=$after"
[ "$after" -gt "$before" ] && echo "PASS" || { echo "FAIL"; exit 1; }
