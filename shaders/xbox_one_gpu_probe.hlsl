// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT
// Small FP32 matrix-vector product without wave ops or packed integer dots.
StructuredBuffer<float> Weights : register(t0);
StructuredBuffer<float> Vector : register(t1);
RWStructuredBuffer<float> Result : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    float sum = 0.0f;
    for (uint column = 0; column < 64; ++column) {
        sum += Weights[id.x * 64 + column] * Vector[column];
    }
    Result[id.x] = sum;
}
