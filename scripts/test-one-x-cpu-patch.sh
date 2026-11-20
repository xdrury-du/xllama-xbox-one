#!/usr/bin/env bash
# Copyright (c) 2026 xllama contributors
# SPDX-License-Identifier: MIT
# Test the optional ggml patch without editing the pinned submodule checkout.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
cmake_bin="${CMAKE_BIN:-cmake}"
doctest_root="${DOCTEST_ROOT:-$repo/build/linux-test/_deps/doctest-src}"
[[ -f "$doctest_root/doctest/doctest.h" ]] || { echo "Set DOCTEST_ROOT to doctest v2.4.11" >&2; exit 1; }
stage="$(mktemp -d "${TMPDIR:-/tmp}/xllama-cpu-patch.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
mkdir "$stage/llama"
git -C "$repo/llama.cpp" archive HEAD | tar -x -C "$stage/llama"
git -C "$stage/llama" apply "$repo/patches/0002-one-x-cpu-experiments.patch"
git -C "$stage/llama" apply "$repo/patches/0003-one-x-weights-paired-dot.patch"
cat > "$stage/CMakeLists.txt" <<'CMAKE'
# Copyright (c) 2026 xllama contributors
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.22)
project(xllama_cpu_patch_test LANGUAGES C CXX)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_AVX2 OFF CACHE BOOL "" FORCE)
set(GGML_FMA OFF CACHE BOOL "" FORCE)
set(GGML_F16C OFF CACHE BOOL "" FORCE)
set(GGML_BMI2 OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
add_subdirectory(llama EXCLUDE_FROM_ALL)
add_executable(test-one-x-cpu-patch "${SMOL_REPO}/tests/test_one_x_cpu_patch.cpp")
target_include_directories(test-one-x-cpu-patch PRIVATE "${SMOL_DOCTEST}")
target_compile_features(test-one-x-cpu-patch PRIVATE cxx_std_17)
target_link_libraries(test-one-x-cpu-patch PRIVATE llama ggml ggml-cpu)
CMAKE
args=()
if [[ "$(uname -s)" == Darwin ]]; then args+=(-DCMAKE_OSX_ARCHITECTURES=x86_64); fi
"$cmake_bin" -S "$stage" -B "$stage/build" -DCMAKE_BUILD_TYPE=Release \
    -DSMOL_REPO="$repo" -DSMOL_DOCTEST="$doctest_root" "${args[@]}"
"$cmake_bin" --build "$stage/build" --target test-one-x-cpu-patch -j "${SMOL_BUILD_JOBS:-2}"
"$stage/build/test-one-x-cpu-patch"
