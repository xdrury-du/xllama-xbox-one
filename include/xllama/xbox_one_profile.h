// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT

// Reviewed model artifacts for the One X experiment. Model identity, download
// origin and digest are an inseparable pin, not independent allowlists.
#pragma once

#include <string_view>

namespace xllama {

struct XboxOneModelPin {
    std::string_view id;
    std::wstring_view base_url;
    std::wstring_view filename;
    std::wstring_view sha256;
};

inline constexpr XboxOneModelPin kXboxOneModels[] = {
    {"gemma3-270m", L"https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main",
     L"gemma-3-270m-it-Q4_K_M.gguf",
     L"b1baabd6b729e4041822220d3e648e00d99cac5df86b10dffb77bcccf0688e39"},
    {"ministral3-3b-instruct",
     L"https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/resolve/"
     L"eb599d408350ea2bb60452cb86be7c7b2fc28227",
     L"Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
     L"9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8"},
};

inline constexpr bool xbox_one_model_allowed(std::string_view id) {
    for (const auto& pin : kXboxOneModels)
        if (id == pin.id)
            return true;
    return false;
}

inline constexpr bool xbox_one_download_allowed(std::wstring_view base_url,
                                                std::wstring_view filename,
                                                std::wstring_view sha256) {
    for (const auto& pin : kXboxOneModels)
        if (base_url == pin.base_url && filename == pin.filename && sha256 == pin.sha256)
            return true;
    return false;
}

} // namespace xllama
