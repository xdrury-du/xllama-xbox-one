// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT
#include "xllama/gpu_probe.h"

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#if !defined(_WIN32) || !defined(XLLAMA_XBOX_ONE)
TEST_CASE("GPU probe cannot report hardware execution on an unsupported host") {
    const auto report = nlohmann::json::parse(xllama::gpu_probe_json());
    CHECK(report.at("schema") == 1);
    CHECK_FALSE(report.at("d3d12_device").get<bool>());
    CHECK_FALSE(report.at("compute_completed").get<bool>());
    CHECK_FALSE(report.at("correct").get<bool>());
    CHECK_FALSE(report.at("model_gpu_offload_enabled").get<bool>());
    CHECK_FALSE(report.at("error").get<std::string>().empty());
}
#endif
