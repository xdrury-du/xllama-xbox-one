// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT
// Built by test-one-x-cpu-patch.sh against an isolated patched ggml checkout.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <doctest/doctest.h>
#include <fstream>
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

TEST_CASE("paired Q4 and Q6 matvec preserve each row including odd tails") {
    for (const auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        for (const int width : {256, 3072, 9216}) {
            for (const int threads : {1, 4, 6}) {
                auto ctx = context();
                REQUIRE(ctx);
                constexpr int rows = 129;
                auto* weights = ggml_new_tensor_2d(ctx.get(), type, width, rows);
                std::vector<float> data(width * rows);
                for (size_t i = 0; i < data.size(); ++i)
                    data[i] = std::sin(static_cast<float>(i) * 0.037f) * 0.2f;
                ggml_quantize_chunk(type, data.data(), weights->data, 0, rows, width, nullptr);
                auto* input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 1);
                for (int i = 0; i < width; ++i)
                    static_cast<float*>(input->data)[i] = std::cos(i * 0.021f);
                auto* output = ggml_mul_mat(ctx.get(), weights, input);
                compute(ctx.get(), output, 0, threads);
                const auto stock = values(output);
                compute(ctx.get(), output, 24, threads);
                const auto paired = values(output);
                CHECK(std::memcmp(stock.data(), paired.data(), stock.size() * sizeof(float)) == 0);
                if (type == GGML_TYPE_Q4_K) {
                    CHECK(ggml_cpu_xllama_pair_q4_calls() > 0);
                    CHECK(ggml_cpu_xllama_pair_q6_calls() == 0);
                } else {
                    CHECK(ggml_cpu_xllama_pair_q6_calls() > 0);
                    CHECK(ggml_cpu_xllama_pair_q4_calls() == 0);
                }
            }
        }
    }
}

#include "gguf.h"
#include "llama.h"
#include <cstdio>
#include <fstream>

TEST_CASE("load-time compression shrinks selected real GGUF buffers without changing source") {
    auto ctx = context();
    REQUIRE(ctx);
    auto* file = gguf_init_empty();
    REQUIRE(file);
    gguf_set_val_str(file, "general.architecture", "llama");
    gguf_set_val_str(file, "tokenizer.ggml.model", "none");
    gguf_set_val_u32(file, "llama.vocab_size", 32);
    gguf_set_val_u32(file, "llama.context_length", 128);
    gguf_set_val_u32(file, "llama.embedding_length", 256);
    gguf_set_val_u32(file, "llama.block_count", 1);
    gguf_set_val_u32(file, "llama.feed_forward_length", 512);
    gguf_set_val_u32(file, "llama.attention.head_count", 8);
    gguf_set_val_u32(file, "llama.attention.head_count_kv", 2);
    gguf_set_val_f32(file, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    const auto add = [&](const char* name, ggml_type type, int64_t cols, int64_t rows) {
        auto* tensor = ggml_new_tensor_2d(ctx.get(), type, cols, rows);
        ggml_set_name(tensor, name);
        std::vector<float> data(cols * rows);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = std::sin(i * 0.037f) * 0.2f;
        if (type == GGML_TYPE_F32)
            std::memcpy(tensor->data, data.data(), data.size() * sizeof(float));
        else
            ggml_quantize_chunk(type, data.data(), tensor->data, 0, rows, cols, nullptr);
        gguf_add_tensor(file, tensor);
    };
    add("token_embd.weight", GGML_TYPE_Q6_K, 256, 32);
    add("output_norm.weight", GGML_TYPE_F32, 256, 1);
    add("blk.0.attn_norm.weight", GGML_TYPE_F32, 256, 1);
    add("blk.0.attn_q.weight", GGML_TYPE_Q4_K, 256, 256);
    add("blk.0.attn_k.weight", GGML_TYPE_Q4_K, 256, 64);
    add("blk.0.attn_v.weight", GGML_TYPE_Q6_K, 256, 64);
    add("blk.0.attn_output.weight", GGML_TYPE_Q4_K, 256, 256);
    add("blk.0.ffn_norm.weight", GGML_TYPE_F32, 256, 1);
    add("blk.0.ffn_gate.weight", GGML_TYPE_Q4_K, 256, 512);
    add("blk.0.ffn_up.weight", GGML_TYPE_Q4_K, 256, 512);
    add("blk.0.ffn_down.weight", GGML_TYPE_Q6_K, 512, 256);
    const char* path = "test_weight_compression_fixture.gguf";
    REQUIRE(gguf_write_to_file(file, path, false));
    gguf_free(file);
    const auto contents = [&] {
        std::ifstream stream(path, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(stream), {});
    };
    const auto source = contents();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_extra_bufts = false;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    uint64_t reference = 0;
    for (unsigned mode = 0; mode <= 3; ++mode) {
        ggml_cpu_xllama_weight_begin(mode);
        auto* model = llama_model_load_from_file(path, params);
        ggml_cpu_xllama_weight_end();
        REQUIRE(model);
        const uint64_t expected = ((mode & 1) ? (64 + 512) * 66 : 0) + ((mode & 2) ? 32 * 66 : 0);
        if (mode == 0)
            reference = llama_model_xllama_resident_bytes(model);
        CHECK(ggml_cpu_xllama_weight_tensors() ==
              unsigned((mode & 1 ? 2 : 0) + (mode & 2 ? 1 : 0)));
        CHECK(ggml_cpu_xllama_weight_saved_bytes() == expected);
        CHECK(llama_model_xllama_resident_bytes(model) == reference - expected);
        CHECK(ggml_cpu_xllama_weight_mode() == 0);
        llama_model_free(model);
        CHECK(contents() == source);
    }
    std::remove(path);
}
