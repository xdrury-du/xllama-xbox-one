// Copyright (c) 2026 xllama contributors
// SPDX-License-Identifier: MIT
#include "xllama/gpu_probe.h"

#include <nlohmann/json.hpp>

#if defined(_WIN32) && defined(XLLAMA_XBOX_ONE)
    #include "d3d12_compute.h"
    #include "xbox_one_gpu_probe_dxbc.h"
    #include "xllama/d3d12_dyn.h"
    #include "xllama/platform.h"
    #include "xllama/utf8_utils.h"

    #include <array>
    #include <chrono>
    #include <cmath>
    #include <cstring>
    #include <dxgi1_4.h>
    #include <stdexcept>

namespace xllama {
namespace {

using d3d12c::ComPtr;
using json = nlohmann::json;
constexpr UINT kWidth = 64;
constexpr UINT64 kOutputBytes = kWidth * sizeof(float);

struct ProbeResources {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12Resource> input, output, readback;
    ComPtr<ID3D12Fence> fence;
};

void require_hr(HRESULT hr, const char* operation) {
    if (FAILED(hr))
        throw std::runtime_error(d3d12c::hr_message(operation, hr));
}

json run_probe() {
    json result = {{"schema", 1},
                   {"scope", "one-small-compute-test-per-process"},
                   {"d3d12_device", false},
                   {"hardware_adapter", false},
                   {"compute_submitted", false},
                   {"compute_completed", false},
                   {"correct", false},
                   {"model_gpu_offload_enabled", false},
                   {"shader_profile", "cs_5_0"},
                   {"matrix_rows", kWidth},
                   {"matrix_columns", kWidth}};
    // Retain the tiny resource set until process exit, including on timeout:
    // submitted GPU work must never reference buffers already destroyed.
    static ProbeResources gpu;
    const auto begin = std::chrono::steady_clock::now();
    std::string stage = "device";
    try {
        std::string error;
        gpu.device = d3d12c::create_device(&error);
        if (!gpu.device)
            throw std::runtime_error(error);
        result["d3d12_device"] = true;
        result["working_set_mb_before"] = working_set_mb();

        stage = "adapter";
        ComPtr<IDXGIFactory4> factory;
        require_hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        ComPtr<IDXGIAdapter1> adapter;
        require_hr(factory->EnumAdapterByLuid(gpu.device->GetAdapterLuid(), IID_PPV_ARGS(&adapter)),
                   "EnumAdapterByLuid");
        DXGI_ADAPTER_DESC1 description = {};
        require_hr(adapter->GetDesc1(&description), "GetDesc1");
        result["adapter"] = wstring_to_utf8(description.Description);
        result["hardware_adapter"] = (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0;
        if (!result["hardware_adapter"].get<bool>())
            throw std::runtime_error("software adapter rejected");

        D3D12_FEATURE_DATA_ARCHITECTURE architecture = {};
        const auto arch_hr = gpu.device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE,
                                                             &architecture, sizeof(architecture));
        result["architecture_query"] = d3d12c::hr_message("CheckFeatureSupport", arch_hr);
        if (SUCCEEDED(arch_hr)) {
            result["uma"] = architecture.UMA != FALSE;
            result["cache_coherent_uma"] = architecture.CacheCoherentUMA != FALSE;
        }
        result["shader_model_queries"] = json::array();
        for (auto requested : {D3D_SHADER_MODEL_6_4, D3D_SHADER_MODEL_6_0, D3D_SHADER_MODEL_5_1}) {
            D3D12_FEATURE_DATA_SHADER_MODEL model = {requested};
            const auto hr =
                gpu.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model));
            result["shader_model_queries"].push_back(
                {{"requested", static_cast<int>(requested)},
                 {"returned", SUCCEEDED(hr) ? static_cast<int>(model.HighestShaderModel) : 0},
                 {"result", d3d12c::hr_message("CheckFeatureSupport", hr)}});
        }
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 options = {};
        const auto wave_hr = gpu.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options,
                                                             sizeof(options));
        result["wave_query"] = d3d12c::hr_message("CheckFeatureSupport", wave_hr);
        if (SUCCEEDED(wave_hr))
            result["wave_ops"] = options.WaveOps != FALSE;
        ComPtr<IDXGIAdapter3> adapter3;
        if (SUCCEEDED(adapter.As(&adapter3))) {
            DXGI_QUERY_VIDEO_MEMORY_INFO memory = {};
            const auto hr =
                adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory);
            result["memory_query"] = d3d12c::hr_message("QueryVideoMemoryInfo", hr);
            if (SUCCEEDED(hr)) {
                result["gpu_budget_bytes"] = memory.Budget;
                result["gpu_usage_bytes"] = memory.CurrentUsage;
            }
        }

        stage = "pipeline";
        D3D12_ROOT_PARAMETER parameters[3] = {};
        for (UINT i = 0; i < 3; ++i) {
            parameters[i].ParameterType =
                i < 2 ? D3D12_ROOT_PARAMETER_TYPE_SRV : D3D12_ROOT_PARAMETER_TYPE_UAV;
            parameters[i].Descriptor.ShaderRegister = i < 2 ? i : 0;
            parameters[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        D3D12_ROOT_SIGNATURE_DESC signature = {};
        signature.NumParameters = 3;
        signature.pParameters = parameters;
        auto serialize = d3d12_dyn::SerializeRootSignature();
        if (!serialize)
            throw std::runtime_error("D3D12SerializeRootSignature unavailable");
        ComPtr<ID3DBlob> blob, errors;
        require_hr(serialize(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors),
                   "D3D12SerializeRootSignature");
        require_hr(gpu.device->CreateRootSignature(0, blob->GetBufferPointer(),
                                                   blob->GetBufferSize(), IID_PPV_ARGS(&gpu.root)),
                   "CreateRootSignature");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline = {};
        pipeline.pRootSignature = gpu.root.Get();
        pipeline.CS = {g_xbox_one_gpu_probe, sizeof(g_xbox_one_gpu_probe)};
        require_hr(gpu.device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&gpu.pipeline)),
                   "CreateComputePipelineState cs_5_0");
        result["pipeline_created"] = true;

        stage = "buffers";
        auto buffer = [&](UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags,
                          D3D12_RESOURCE_STATES state, const char* name) {
            auto value =
                d3d12c::create_buffer(gpu.device.Get(), bytes, heap, flags, state, name, &error);
            if (!value)
                throw std::runtime_error(error);
            return value;
        };
        std::array<float, kWidth * kWidth + kWidth> input;
        std::array<float, kWidth> expected = {};
        for (UINT column = 0; column < kWidth; ++column)
            input[kWidth * kWidth + column] = static_cast<float>(static_cast<int>(column % 7) - 3);
        for (UINT row = 0; row < kWidth; ++row) {
            for (UINT column = 0; column < kWidth; ++column) {
                const float weight =
                    static_cast<float>(static_cast<int>((row * 7 + column * 3) % 11) - 5);
                input[row * kWidth + column] = weight;
                expected[row] += weight * input[kWidth * kWidth + column];
            }
        }
        gpu.input = buffer(sizeof(input), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_GENERIC_READ, "probe input");
        gpu.output = buffer(kOutputBytes, D3D12_HEAP_TYPE_DEFAULT,
                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "probe output");
        gpu.readback = buffer(kOutputBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                              D3D12_RESOURCE_STATE_COPY_DEST, "probe readback");
        void* mapped = nullptr;
        D3D12_RANGE empty = {0, 0};
        require_hr(gpu.input->Map(0, &empty, &mapped), "Map input");
        std::memcpy(mapped, input.data(), sizeof(input));
        gpu.input->Unmap(0, nullptr);

        stage = "dispatch";
        D3D12_COMMAND_QUEUE_DESC queue = {};
        queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        require_hr(gpu.device->CreateCommandQueue(&queue, IID_PPV_ARGS(&gpu.queue)),
                   "CreateCommandQueue");
        require_hr(gpu.device->CreateCommandAllocator(queue.Type, IID_PPV_ARGS(&gpu.allocator)),
                   "CreateCommandAllocator");
        require_hr(gpu.device->CreateCommandList(0, queue.Type, gpu.allocator.Get(),
                                                 gpu.pipeline.Get(), IID_PPV_ARGS(&gpu.commands)),
                   "CreateCommandList");
        gpu.commands->SetComputeRootSignature(gpu.root.Get());
        gpu.commands->SetComputeRootShaderResourceView(0, gpu.input->GetGPUVirtualAddress());
        gpu.commands->SetComputeRootShaderResourceView(1, gpu.input->GetGPUVirtualAddress() +
                                                              kWidth * kWidth * sizeof(float));
        gpu.commands->SetComputeRootUnorderedAccessView(2, gpu.output->GetGPUVirtualAddress());
        gpu.commands->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = gpu.output.Get();
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        gpu.commands->ResourceBarrier(1, &barrier);
        gpu.commands->CopyResource(gpu.readback.Get(), gpu.output.Get());
        require_hr(gpu.commands->Close(), "Close command list");
        require_hr(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gpu.fence)),
                   "CreateFence");
        ID3D12CommandList* lists[] = {gpu.commands.Get()};
        gpu.queue->ExecuteCommandLists(1, lists);
        result["compute_submitted"] = true;
        require_hr(gpu.queue->Signal(gpu.fence.Get(), 1), "Signal");
        stage = "fence";
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            const auto completed = gpu.fence->GetCompletedValue();
            if (completed == UINT64_MAX)
                throw std::runtime_error(
                    d3d12c::hr_message("device removed", gpu.device->GetDeviceRemovedReason()));
            if (completed >= 1)
                break;
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error(
                    "GPU fence timed out after 5 seconds; no further probe this process");
            Sleep(1);
        }
        result["compute_completed"] = true;

        stage = "readback";
        D3D12_RANGE read_range = {0, static_cast<SIZE_T>(kOutputBytes)};
        require_hr(gpu.readback->Map(0, &read_range, &mapped), "Map readback");
        std::array<float, kWidth> actual;
        std::memcpy(actual.data(), mapped, sizeof(actual));
        gpu.readback->Unmap(0, &empty);
        UINT mismatches = 0;
        for (UINT row = 0; row < kWidth; ++row)
            if (!std::isfinite(actual[row]) || actual[row] != expected[row])
                ++mismatches;
        result["mismatched_rows"] = mismatches;
        result["correct"] = mismatches == 0;
        result["working_set_mb_after"] = working_set_mb();
        stage = "complete";
    } catch (const std::exception& error) {
        result["error"] = error.what();
    } catch (...) {
        result["error"] = "unexpected GPU probe exception";
    }
    result["stage"] = stage;
    result["elapsed_ms"] =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    return result;
}

} // namespace

std::string gpu_probe_json() {
    static const std::string report = run_probe().dump();
    return report;
}

} // namespace xllama
#else
namespace xllama {
std::string gpu_probe_json() {
    return nlohmann::json({{"schema", 1},
                           {"d3d12_device", false},
                           {"compute_completed", false},
                           {"correct", false},
                           {"model_gpu_offload_enabled", false},
                           {"error", "GPU probe requires the Xbox One Windows build"}})
        .dump();
}
} // namespace xllama
#endif
