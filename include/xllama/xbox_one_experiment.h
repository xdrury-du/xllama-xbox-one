// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "xllama/session.h"

#include <string_view>

namespace xllama {

// Every candidate changes one knob from the same long-context CPU baseline.
// Bounds are reviewed here rather than accepting arbitrary load parameters.
inline bool apply_xbox_one_experiment(std::string_view model, std::string_view id,
                                      SessionParams& params) {
    if (id != "baseline" && id != "gpu-1" && id != "gpu-8" && id != "gpu-27" && id != "threads-4" &&
        id != "threads-2" && id != "ubatch-64" && id != "ubatch-256" && id != "kv-q8" &&
        id != "prompt-lookup")
        return false;
    if (model != "ministral3-3b-instruct")
        return false;
    params.n_ctx = 8192;
    params.n_threads = 6;
    params.n_batch = 512;
    params.n_ubatch = 128;
    params.n_gpu_layers = 0;
    params.kv_q8 = false;
    params.prompt_lookup = false;
    if (id == "gpu-1")
        params.n_gpu_layers = 1;
    if (id == "gpu-8")
        params.n_gpu_layers = 8;
    if (id == "gpu-27")
        params.n_gpu_layers = 27;
    if (id == "threads-4")
        params.n_threads = 4;
    if (id == "threads-2")
        params.n_threads = 2;
    if (id == "ubatch-64")
        params.n_ubatch = 64;
    if (id == "ubatch-256")
        params.n_ubatch = 256;
    if (id == "kv-q8")
        params.kv_q8 = true;
    if (id == "prompt-lookup")
        params.prompt_lookup = true;
    return true;
}

} // namespace xllama
