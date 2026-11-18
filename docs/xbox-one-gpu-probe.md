# Xbox One X GPU access probe

This diagnostic establishes hardware GPU access independently of model
offload. The One X build retains its AVX CPU inference profile and exact
two-model catalogue. No model weights are converted or sent to the GPU.

## Run

Install the newer One X package in place. On Xbox Dev Home set its App type
to Game, fully relaunch xllama and keep it foregrounded with LAN API enabled.
After startup model loading completes, call:

```bash
curl --max-time 60 http://<console-ip>:11434/v1/diagnostics/gpu \
  -H 'Content-Type: application/json' -d '{}'
```

The route exists only in the One X build. It uses the same process-wide
compute admission as generation and returns busy if another job owns that
slot. The result is cached once per process. Relaunch to run it again.

## What is measured

The probe creates a system Direct3D 12 device at feature level 11_0 and
rejects a software adapter. It reports adapter identity, UMA/cache coherence,
shader model queries, wave-operation support and the local GPU memory budget
where those queries succeed. A failed optional query stays an explicit
HRESULT rather than becoming a false unsupported verdict.

It then creates a pipeline from an SDK-FXC-compiled `cs_5_0` DXBC shader,
uploads a 64 by 64 FP32 matrix and 64-element vector, dispatches one group,
copies 64 results back and compares every element against a CPU reference.
The inputs are small integers exactly representable throughout the sum, so
exact equality is appropriate. No wave operations, packed integer dot
products, custom shared-memory heaps or inference shaders are required.

The fence wait is bounded to five seconds. Resources remain alive until
process exit, including on timeout, so submitted work cannot reference freed
buffers. Only one small resource set is retained. A timeout is not retried
inside that process. Driver API calls themselves have no application-level
timeout guarantee.

## Interpret

`hardware_adapter`, `compute_submitted`, `compute_completed` and `correct`
must all be true to claim successful hardware compute. A device or shader
query alone does not establish it. `stage` and `error` identify the first
required failure. Capability-query failures are recorded separately.

`elapsed_ms` includes initialization and synchronization. It is not a GPU
kernel timing, bandwidth measurement or model-throughput prediction. A pass
does not validate the existing Series SM 6.4 inference shaders, the backend's
cache-coherent-UMA requirement, Ministral numerics or a speed improvement.
Those require subsequent experiments.

## Build and validation

The regular One X build command compiles the probe shader with the installed
Windows SDK `fxc.exe` before MSBuild. Its generated header is not committed.
Linux tests check that the host stub cannot report GPU success. Windows
compilation checks the real implementation, and the console endpoint is the
hardware execution gate. See the [CPU runbook](xbox-one-experiment.md) for
the build and model restrictions.
