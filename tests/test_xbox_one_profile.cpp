// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT

#include "xllama/xbox_one_profile.h"

#include <doctest/doctest.h>
#include <string>

using namespace xllama;

TEST_CASE("Xbox One profile: only the reviewed model id is admitted") {
    CHECK(xbox_one_model_allowed("gemma3-270m"));
    for (const auto* id : {"", "qwen35-0.8b", "lfm25-350m", "gemma4-e2b", "Gemma3-270m",
                           "gemma3-270m-other", "../gemma3-270m", "C:\\models\\gemma3-270m"}) {
        CAPTURE(id);
        CHECK_FALSE(xbox_one_model_allowed(id));
    }
}

TEST_CASE("Xbox One profile: download requires the exact source, file and digest") {
    CHECK(xbox_one_download_allowed(kXboxOneModelBaseUrl, kXboxOneModelFilename,
                                    kXboxOneModelSha256));
    CHECK_FALSE(xbox_one_download_allowed(L"https://example.org/model", kXboxOneModelFilename,
                                          kXboxOneModelSha256));
    CHECK_FALSE(
        xbox_one_download_allowed(kXboxOneModelBaseUrl, L"other.gguf", kXboxOneModelSha256));
    CHECK_FALSE(xbox_one_download_allowed(kXboxOneModelBaseUrl, kXboxOneModelFilename, L""));
    CHECK_FALSE(xbox_one_download_allowed(kXboxOneModelBaseUrl, kXboxOneModelFilename,
                                          std::wstring(64, L'0')));
}
