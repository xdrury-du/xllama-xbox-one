// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT

#include "xllama/xbox_one_profile.h"

#include <algorithm>
#include <doctest/doctest.h>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>

using namespace xllama;

TEST_CASE("Xbox One profile: only the reviewed model id is admitted") {
    CHECK(xbox_one_model_allowed("gemma3-270m"));
    CHECK(xbox_one_model_allowed("ministral3-3b-instruct"));
    for (const auto* id : {"", "qwen35-0.8b", "lfm25-350m", "gemma4-e2b", "Gemma3-270m",
                           "gemma3-270m-other", "../gemma3-270m", "C:\\models\\gemma3-270m"}) {
        CAPTURE(id);
        CHECK_FALSE(xbox_one_model_allowed(id));
    }
}

TEST_CASE("Xbox One profile: download pins cannot be mixed across models") {
    for (const auto& pin : kXboxOneModels) {
        CAPTURE(pin.id);
        CHECK(xbox_one_download_allowed(pin.base_url, pin.filename, pin.sha256));
        CHECK_FALSE(
            xbox_one_download_allowed(L"https://example.org/model", pin.filename, pin.sha256));
        CHECK_FALSE(xbox_one_download_allowed(pin.base_url, L"other.gguf", pin.sha256));
        CHECK_FALSE(xbox_one_download_allowed(pin.base_url, pin.filename, L""));
        CHECK_FALSE(xbox_one_download_allowed(pin.base_url, pin.filename, std::wstring(64, L'0')));
    }
    const auto& gemma = kXboxOneModels[0];
    const auto& ministral = kXboxOneModels[1];
    CHECK_FALSE(xbox_one_download_allowed(ministral.base_url, gemma.filename, gemma.sha256));
    CHECK_FALSE(xbox_one_download_allowed(ministral.base_url, ministral.filename, gemma.sha256));
    CHECK_FALSE(xbox_one_download_allowed(
        ministral.base_url, L"Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf", ministral.sha256));
}

TEST_CASE("Xbox One profile: packaged catalogue agrees with reviewed text artifacts") {
    std::ifstream input(XBOX_ONE_CATALOGUE_PATH);
    REQUIRE(input.good());
    const auto catalogue = nlohmann::json::parse(input);
    REQUIRE(catalogue.at("models").size() == std::size(kXboxOneModels));
    for (const auto& pin : kXboxOneModels) {
        const auto& entries = catalogue.at("models");
        auto entry = std::find_if(entries.begin(), entries.end(), [&](const auto& value) {
            return value.at("name").template get<std::string>() == pin.id;
        });
        REQUIRE(entry != entries.end());
        REQUIRE(entry->at("files").size() == 1);
        const auto wide = [](const std::string& text) {
            return std::wstring(text.begin(), text.end());
        };
        const auto& file = entry->at("files").at(0);
        CHECK(xbox_one_download_allowed(wide(entry->at("hf_base_url").get<std::string>()),
                                        wide(file.at("filename").get<std::string>()),
                                        wide(file.at("sha256").get<std::string>())));
        CHECK(entry->at("kind") == "gguf");
        CHECK(entry->at("gpu_layers") == 0);
        CHECK(entry->at("n_ctx") == 1024);
        CHECK(entry->at("n_predict") == 128);
        CHECK(file.at("filename").get<std::string>().find("mmproj") == std::string::npos);
    }
}
