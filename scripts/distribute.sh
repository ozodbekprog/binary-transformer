#!/usr/bin/env bash
# Uchta node'ga parallel tarqatish va benchmark
set -u
CR="python3 $HOME/.local/bin/centerrun"
D="/tmp/opencode/bt-dist"

RYZEN_CMD='mkdir -p ~/bt/bin && cp benchmark_engine binary_matmul.spv ~/bt/bin/ && chmod +x ~/bt/bin/benchmark_engine && echo "libvulkan: $(ldconfig -p 2>/dev/null | grep -m1 libvulkan || echo YOQ)" && ~/bt/bin/benchmark_engine --check --tokens 3 2>&1 | tail -16'

BUILD_CMD='mkdir -p ~/bt && tar xzf binary-transformer.tgz -C ~/bt && cd ~/bt/binary-transformer && bash scripts/build_simple.sh 2>&1 | tail -3 && ./bin/bt_selftest 2>&1 | tail -1 && ./bin/benchmark_engine --check --tokens 3 2>&1 | tail -14'

$CR "$RYZEN_CMD" ryzen-ai7 400 --file "$D/benchmark_engine" --file "$D/binary_matmul.spv" > /tmp/opencode/bt_ryzen.txt 2>&1 &
P1=$!
$CR "$BUILD_CMD" asror-fedora 600 --file "$D/binary-transformer.tgz" > /tmp/opencode/bt_asror.txt 2>&1 &
P2=$!
$CR "CXX=clang++ $BUILD_CMD" "Ozodbeks-MBP" 600 --file "$D/binary-transformer.tgz" > /tmp/opencode/bt_mac.txt 2>&1 &
P3=$!
wait $P1 $P2 $P3
echo "=== RYZEN ==="; cat /tmp/opencode/bt_ryzen.txt
echo "=== ASROR ==="; cat /tmp/opencode/bt_asror.txt
echo "=== MAC ==="; cat /tmp/opencode/bt_mac.txt
