// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT

// Force-included in every app, llama-common and ggml translation unit.
#pragma once

#if !defined(XLLAMA_XBOX_ONE)
    #error "Xbox One build profile was not propagated to this project"
#endif

#if defined(__AVX2__) || defined(__AVX512F__) || defined(__FMA__) || defined(__BMI2__)
    #error "Xbox One Jaguar does not support AVX2, AVX-512, FMA3 or BMI2"
#endif
