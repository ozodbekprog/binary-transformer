#!/usr/bin/env bash
# build_simple.sh — CMake'siz tez yig'ish (Mac/Linux/CMake bo'lmagan node'lar).
# Vulkan header topilsa GPU bilan, aks holda CPU-only build.
set -euo pipefail
cd "$(dirname "$0")/.."

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null 2>&1; then
  CXX="clang++"
fi
if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "XATO: C++ kompilyator topilmadi (g++ yoki clang++)" >&2
  exit 1
fi

mkdir -p bin
CXXFLAGS="-std=c++20 -O3 -pthread -Wall -Wextra -I. -Isrc"

VKFLAG="-DBT_HAVE_VULKAN=0"
if [ -f /usr/include/vulkan/vulkan.h ] || \
   [ -f /opt/homebrew/include/vulkan/vulkan.h ] || \
   [ -n "${BT_VULKAN_HEADER_DIR:-}" ]; then
  VKFLAG="-DBT_HAVE_VULKAN=1"
  [ -n "${BT_VULKAN_HEADER_DIR:-}" ] && CXXFLAGS="$CXXFLAGS -I${BT_VULKAN_HEADER_DIR}"
fi
echo "== CXX: $CXX | Vulkan: $VKFLAG"

echo "== benchmark_engine"
$CXX $CXXFLAGS $VKFLAG \
  src/cli/benchmark_engine.cpp src/core/ternary_simd.cpp src/gpu/vulkan_wrapper.cpp \
  -o bin/benchmark_engine

echo "== pipeline_ping"
$CXX $CXXFLAGS $VKFLAG src/cli/pipeline_ping.cpp -o bin/pipeline_ping

echo "== distributed_70b_engine"
$CXX $CXXFLAGS $VKFLAG \
  src/main_multi_node.cpp src/core/ternary_simd.cpp src/gpu/vulkan_wrapper.cpp \
  -o bin/distributed_70b_engine

echo "== bt_selftest"
$CXX $CXXFLAGS $VKFLAG tests/selftest.cpp src/core/ternary_simd.cpp -o bin/bt_selftest

if command -v glslc >/dev/null 2>&1; then
  glslc src/gpu/binary_matmul.comp -o bin/binary_matmul.spv
  echo "SPIR-V: bin/binary_matmul.spv"
elif command -v glslangValidator >/dev/null 2>&1; then
  glslangValidator -V src/gpu/binary_matmul.comp -o bin/binary_matmul.spv
  echo "SPIR-V: bin/binary_matmul.spv"
else
  echo "OGOHLANTIRISH: glslc/glslangValidator yo'q — GPU benchmark ishlamaydi (CPU ishlaydi)"
fi

echo "Tayyor: $(pwd)/bin/"
