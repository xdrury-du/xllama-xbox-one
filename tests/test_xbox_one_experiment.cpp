// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT
#include "xllama/xbox_one_experiment.h"

#include <doctest/doctest.h>

TEST_CASE("One X experiment profiles change exactly one baseline setting") {
    xllama::SessionParams baseline;
    baseline.model_path = "resolved/model/directory";
    REQUIRE(xllama::apply_xbox_one_experiment("ministral3-3b-instruct", "baseline", baseline));
    for (const auto* id : {"gpu-1", "gpu-8", "gpu-27", "threads-4", "threads-2", "ubatch-64",
                           "ubatch-256", "kv-q8", "prompt-lookup"}) {
        SUBCASE(id) {
            auto candidate = baseline;
            REQUIRE(xllama::apply_xbox_one_experiment("ministral3-3b-instruct", id, candidate));
            CHECK(candidate.model_path == baseline.model_path);
            CHECK(candidate.n_ctx == 8192);
            CHECK(candidate.n_batch == 512);
            const int differences = (candidate.n_threads != baseline.n_threads) +
                                    (candidate.n_ubatch != baseline.n_ubatch) +
                                    (candidate.n_gpu_layers != baseline.n_gpu_layers) +
                                    (candidate.kv_q8 != baseline.kv_q8) +
                                    (candidate.prompt_lookup != baseline.prompt_lookup);
            CHECK(differences == 1);
        }
    }
}

TEST_CASE("One X experiment rejects other model ids and unknown profiles without mutation") {
    xllama::SessionParams params;
    params.n_ctx = 1234;
    CHECK_FALSE(xllama::apply_xbox_one_experiment("gemma3-270m", "gpu-27", params));
    CHECK_FALSE(xllama::apply_xbox_one_experiment("ministral3-3b-instruct", "arbitrary", params));
    CHECK(params.n_ctx == 1234);
}
