// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT
// Built by test-one-x-cpu-patch.sh against an isolated patched ggml checkout.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <doctest/doctest.h>
#include <memory>
#include <vector>

namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
Context context() {
    return Context(ggml_init({32 * 1024 * 1024, nullptr, false}), ggml_free);
}
void compute(ggml_context* ctx, ggml_tensor* output, unsigned flags, int threads) {
    ggml_cpu_xllama_set_flags(flags);
    ggml_cpu_xllama_reset_stats();
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    REQUIRE(ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS);
}
std::vector<float> values(const ggml_tensor* tensor) {
    const auto* data = static_cast<const float*>(tensor->data);
    return {data, data + ggml_nelements(tensor)};
}
} // namespace

TEST_CASE("guided matmul covers rows and odd tails with identical arithmetic") {
    for (const auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        for (const int rows : {129, 257, 1031}) {
            for (const int threads : {1, 4, 6}) {
                auto ctx = context();
                REQUIRE(ctx);
                constexpr int width = 3072;
                auto* weights = ggml_new_tensor_2d(ctx.get(), type, width, rows);
                std::vector<float> data(width * rows);
                for (size_t i = 0; i < data.size(); ++i)
                    data[i] = std::sin(static_cast<float>(i) * 0.037f) * 0.2f;
                ggml_quantize_chunk(type, data.data(), weights->data, 0, rows, width, nullptr);
                auto* input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 1);
                for (int i = 0; i < width; ++i)
                    static_cast<float*>(input->data)[i] = std::cos(i * 0.021f);
                auto* output = ggml_mul_mat(ctx.get(), weights, input);
                ggml_set_name(output, "result_output");
                compute(ctx.get(), output, 1, threads);
                const auto reference = values(output);
                compute(ctx.get(), output, 3, threads);
                const auto candidate = values(output);
                CHECK(std::memcmp(reference.data(), candidate.data(),
                                  reference.size() * sizeof(float)) == 0);
                CHECK(ggml_cpu_xllama_guided_calls() == 1);
                const auto stat = ggml_cpu_xllama_op_stat(GGML_OP_MUL_MAT);
                CHECK(stat.calls == 1);
                CHECK(stat.wall_us > 0);
                CHECK(stat.head_wall_us > 0);
            }
        }
    }
}

TEST_CASE("single-row binary and silu paths cover uneven column partitions") {
    for (const int width : {1025, 3072, 9216}) {
        for (const int threads : {1, 4, 6}) {
            for (const int kind : {0, 1, 2}) {
                auto ctx = context();
                REQUIRE(ctx);
                auto* a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
                auto* b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
                for (int i = 0; i < width; ++i) {
                    static_cast<float*>(a->data)[i] = std::sin(i * 0.01f);
                    static_cast<float*>(b->data)[i] = std::cos(i * 0.02f);
                }
                auto* out = kind == 0   ? ggml_add(ctx.get(), a, b)
                            : kind == 1 ? ggml_mul(ctx.get(), a, b)
                                        : ggml_silu(ctx.get(), a);
                compute(ctx.get(), out, 0, threads);
                const auto reference = values(out);
                compute(ctx.get(), out, 4, threads);
                const auto candidate = values(out);
                float max_error = 0;
                for (size_t i = 0; i < reference.size(); ++i)
                    max_error = std::max(max_error, std::abs(reference[i] - candidate[i]));
                CHECK(max_error <= 1e-6f);
                CHECK(ggml_cpu_xllama_row_calls() == 1);
            }
        }
    }
}

TEST_CASE("multi-row tensors retain the stock path and flags reset statistics") {
    auto ctx = context();
    auto* a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3072, 2);
    auto* b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3072, 2);
    ggml_set_f32(a, 2.0f);
    ggml_set_f32(b, 3.0f);
    auto* out = ggml_add(ctx.get(), a, b);
    compute(ctx.get(), out, 7, 6);
    CHECK(ggml_cpu_xllama_row_calls() == 0);
    for (const auto x : values(out))
        CHECK(x == 5.0f);
    ggml_cpu_xllama_reset_stats();
    CHECK(ggml_cpu_xllama_op_stat(GGML_OP_ADD).calls == 0);
    ggml_cpu_xllama_set_flags(0);
    CHECK(ggml_cpu_xllama_flags() == 0);
}

TEST_CASE("guided mode preserves multi-column prefill and scalar broadcast fallback") {
    auto ctx = context();
    auto* weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, 257);
    auto* input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, 2);
    ggml_set_f32(weights, 0.5f);
    ggml_set_f32(input, 0.25f);
    auto* out = ggml_mul_mat(ctx.get(), weights, input);
    compute(ctx.get(), out, 2, 6);
    CHECK(ggml_cpu_xllama_guided_calls() == 0);
    for (const auto value : values(out))
        CHECK(value == 32.0f);
    auto* a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3072);
    auto* b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_set_f32(a, 2.0f);
    ggml_set_f32(b, 3.0f);
    auto* sum = ggml_add(ctx.get(), a, b);
    compute(ctx.get(), sum, 4, 6);
    CHECK(ggml_cpu_xllama_row_calls() == 0);
    for (const auto value : values(sum))
        CHECK(value == 5.0f);
    ggml_cpu_xllama_set_flags(0);
}
