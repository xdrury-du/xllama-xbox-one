// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <string>

namespace xllama {

// One cached, small hardware compute test per process. JSON reports capability
// queries separately from verified execution; it never enables model offload.
std::string gpu_probe_json();

} // namespace xllama
