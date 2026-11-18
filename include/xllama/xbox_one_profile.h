// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT

// The initial One X experiment admits one pinned Google model. Extend this
// policy and the experiment catalogue together after provenance review.
#pragma once

#include <string_view>

namespace xllama {

inline constexpr std::wstring_view kXboxOneModelBaseUrl =
    L"https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main";
inline constexpr std::wstring_view kXboxOneModelFilename = L"gemma-3-270m-it-Q4_K_M.gguf";
inline constexpr std::wstring_view kXboxOneModelSha256 =
    L"b1baabd6b729e4041822220d3e648e00d99cac5df86b10dffb77bcccf0688e39";

inline constexpr bool xbox_one_model_allowed(std::string_view id) {
    return id == "gemma3-270m";
}

inline constexpr bool xbox_one_download_allowed(std::wstring_view base_url,
                                                std::wstring_view filename,
                                                std::wstring_view sha256) {
    return base_url == kXboxOneModelBaseUrl && filename == kXboxOneModelFilename &&
           sha256 == kXboxOneModelSha256;
}

} // namespace xllama
