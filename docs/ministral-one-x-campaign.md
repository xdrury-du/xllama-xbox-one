# Ministral One X experiment protocol

This campaign uses the reviewed official Ministral 3 3B Instruct 2512 Q4_K_M
language artifact on the operator-declared Xbox One X. Model revision and
SHA-256 remain the pins in `xbox_one_profile.h`. No additional model lineage
is admitted. Windows builds and Mac tests are preparation, never Xbox results.

## Controls and candidates

All profiles use context 8192, logical batch 512, six CPU threads, physical
microbatch 128, F16 KV, prompt lookup off and zero GPU layers unless the
candidate changes that single field. Sampling is greedy with fixed seed.
Every measured prompt starts with fresh KV while retaining model weights.
Load time, engine prefill, engine decode, first text over the LAN and total
request time are separate measurements. Engine decode includes stream
callbacks. Memory is the maximum observed working set sampled every 20 ms
during the request, including model reload when it occurs, not an exact
allocation peak or an additional VRAM allowance.
New builds also measure cumulative token-callback duration, including stop
filtering and SSE writes. This is a subset of decode-loop time, not an isolated
network round-trip measurement or a pure model-evaluation timer. Callback call
counts are not assumed to equal generated-token counts.

| Profile       | Change from baseline            | Purpose                                              |
| ------------- | ------------------------------- | ---------------------------------------------------- |
| baseline      | None                            | Long-context CPU reference                           |
| gpu-1         | GPU layers 0 to 1               | Conservative offload correctness and overhead        |
| gpu-8         | GPU layers 0 to 8               | Partial-offload scaling                              |
| gpu-27        | GPU layers 0 to 27              | Request all 26 repeating layers and output placement |
| threads-4     | CPU threads 6 to 4              | CPU contention tradeoff                              |
| threads-2     | CPU threads 6 to 2              | Smaller worker pool                                  |
| ubatch-64     | Physical microbatch 128 to 64   | Prefill memory versus speed                          |
| ubatch-256    | Physical microbatch 128 to 256  | Larger prefill work units                            |
| kv-q8         | F16 KV to Q8                    | Cache memory versus speed and answer quality         |
| prompt-lookup | Enable draft-free prompt lookup | Speculation acceptance versus overhead               |

These are compatible experiments informed by smolmaxxing's kernel,
profiling and memory approach. They are not a port of every smolmaxxing
patch. Expert paging does not apply to this dense checkpoint. One-bit
weights and vocabulary pruning are separate model-modification projects,
not implicit effects of enabling this backend.

## GPU gate

One X inference shaders compile as SM5.1 DXBC. Signed-byte multiply/adds
replace SM6.4 packed-dot intrinsics. CPU activation quantization and ggml
tensor layouts stay shared with the existing backend. Before model offload,
every backend selftest must pass its predeclared CPU-reference tolerance,
including Ministral-width 3072 cases, odd output-row tails and multi-column
prefill. The selftest unloads the resident model to avoid overlapping large
test fixtures with model memory. CPU fallback is an explicit experiment
failure. A positive configured layer count alone is insufficient: generation
must report actual GPU dispatches. Other operations still execute on CPU.

## Workloads and sequence

Run every test serially. First check short exact answers, arithmetic,
structured extraction and conversation recall. Preserve failed answers.
Then request a complete twelve-section guide targeting at least 1100 words,
with all section headings and a final marker. Do not pass a reply-token
limit. Generation ends naturally or at available context; record the finish
reason and token count. Responses under 1024 output tokens do not satisfy
the long-response gate and require a clearly recorded follow-up, not silent
replacement of the first outcome.

Record one initial long run for every admitted profile. Reports from a
single run are exploratory observations. Repeat the CPU reference and the
best passing candidates to check reproducibility before recommending a
winner. A final additional workload requests at least 2048 output tokens
from the reference and the best usable candidate to inspect long-response
stability. A stalled, disconnected, out-of-memory or context-limited run is
an outcome, not a completed answer. The pilot used a 45-minute elapsed guard
and completed before it. The main collector has no elapsed request guard by
default, because the measured pilot latency makes a much longer complete
manual plausible. Physical context and a 180-second socket inactivity timeout
still bound generation or detect a stalled connection. An optional
`--request-watchdog-minutes` limit must be recorded and any abort attributed to
that configured guard, not silently classified as a model failure.

## Verdict and reports

### Protocol revision 2, declared after the CPU pilot and before candidates

The initial pilot completed the guide but wrapped its terminal marker in bold
Markdown and returned the requested JSON inside a code fence. Its original
strict-format failures remain preserved. For the main campaign, assess
functional content and tracked raw-format compliance separately. JSON may be
unwrapped from one balanced JSON code fence before validating the exact object
and integer type. The guide marker must be on the terminal line, allowing
Markdown emphasis decoration. Raw JSON and an undecorated exact terminal line
are still reported as separate format checks. A candidate must preserve every
tracked format check that its matched baseline passes. A known baseline format
limitation can persist, but cannot disappear from the report.
The main functional word minimum is raised from the pilot's permissive 700
words to the prompt's requested 1100. Reports also record whether initial
answers match the reference text. Different output content and length can
affect average decode throughput even with the same prompt, so those results
remain workload observations rather than isolated kernel-speed measurements.

This prospective rubric change does not change the GPU numerical tolerance,
performance thresholds, prompts or timing method. It prevents cosmetic
decoration from hiding a measured performance difference while retaining the
information needed for clients that require raw machine-readable output.

Each report includes the exact settings and implementation change, raw
outputs, engine and request timing, output length, sampled RSS, actual GPU
activity, quality checks, errors, a comparison table and standalone graphs.
Plot measured decode throughput separately from prefill/first-text latency.
Output-progress curves use received characters, not an assumed one-token
per SSE chunk conversion.

`Improved = true` requires a successful long response, passing functional
correctness checks, no tracked formatting regression against the matched
baseline and at least 5 percent higher measured decode throughput than the
matched CPU reference. A memory-targeted Q8 KV candidate may instead pass
with at least 10 percent less sampled working set, at most 5 percent decode
regression and no failed quality check. Microbatch candidates may pass with
at least 10 percent faster prefill at the same prefill-token count and no
more than 5 percent decode regression. The smaller microbatch may instead
pass the same 10 percent memory-reduction criterion. Otherwise the value is false, with
status distinguishing regression, failure and improvement not demonstrated.
A single-run true value is explicitly exploratory until repeated. Basic
checks and manual coherence review do not establish general model quality.
Do not describe an unrun candidate as measured or combine improvements
before reporting their individual results.

## User interface

The Mac client exposes the reviewed profiles and accepts a blank reply
budget. It removes the old 512-token client cap. The normal controller UI
and its default catalogue context are separate from these API profiles.
Keep the app foregrounded and confirm App type Game after every install.

## Collector

Run `scripts/xbox-campaign.py --xbox <private-IP> --output <directory> --build-info <artifact-build-info.json>`
with Python 3.9 or newer and matplotlib available. The build receipt is required
and stored with every request. The default sequence runs all profiles serially,
then two additional reference and fastest usable candidate repetitions,
optionally the best memory-targeted winner, followed by the extended manual.
If no candidate passes every short check, a candidate with a successful primary
long response can still be selected for diagnostic repetition and stress.
This does not weaken the strict quality gate or make it a recommendation.
The reference and confirmation candidates alternate to expose time drift.
Before switching profiles, the collector calls the cached GPU-selftest endpoint,
which unloads the resident session, then performs an unmeasured exact-profile
warm-up. This is required for package 1.7.0.7 because its shared session identity
omits thread count, microbatch and prompt lookup. Reusing a resident session
would otherwise retain those previous tuning values. Warm-ups always execute,
including when other request evidence resumes from disk. A failed GPU gate
blocks GPU candidates while still allowing the independent CPU experiments.
Report repeated throughput as the median of individual rates. These are small
samples, not statistical proof. Individual request records remain available.

`--initial-only` stops after the exploratory sweep. `--profiles` narrows the
candidate list but always includes the baseline. Completed requests resume only
when the full payload and build receipt match. Replaced or failed attempts are
preserved. Raw streams, JSON, per-profile HTML tables, PNG/SVG graphs, full
responses and CSV/JSON summaries are written as tests finish. Missing metrics
are shown as unmeasured. A shorter primary reply triggers a separate length
follow-up without replacing the primary comparison or its failed length gate.

## Short kernel campaign

The current candidate entry point is `scripts/xbox-smol-short.py`. The original
long-response campaign above is historical and is not invoked by this entry point.
Every performance request has an explicit 64-token reply cap. Four functional
checks use 8 or 32 tokens, including warm-ups. Three performance repetitions
use the same short bicycle prompt, greedy sampling, fresh KV, warm weights,
8192 context, six threads, batch 512, microbatch 128 and F16 KV. A 180-second
request watchdog and socket timeout guard a stalled request. Runs remain serial.

| Profile      | Effective CPU change                                                  |
| ------------ | --------------------------------------------------------------------- |
| smol-base    | Patched build with all optional CPU changes off                       |
| smol-guided  | Atomically claimed matrix row chunks shrink toward the tail           |
| smol-row     | Split contiguous single-row F32 binary operations and SiLU by columns |
| smol-both    | Combine the two previously isolated changes                           |
| smol-profile | Instrument stock operations, separate from performance comparisons    |

These adapt the guided scheduling and single-row work sharing described in
[smolmaxxing](https://github.com/hyperionaidev/smolmaxxing/blob/main/docs/how-it-works.md)
to this pin and the Xbox CPU. No ARM intrinsics or trained model weights are imported.
The output vocabulary already has 131072 rows, divisible by 32. The greedy sampler
already scans logits without sorting. Those observations are admission checks,
not new optimizations or measured speedups. Vocabulary pruning, weight
requantization and custom attention remain separate, unimplemented candidates.

The new API profiles reject omitted reply limits or limits above 64 before model
loading. CPU flags are process-global only while the serialized API generation
owns the compute permit and session lock. A scope guard resets flags to zero on
return or exception, so the normal GUI does not inherit the experiment. Profile
telemetry reports accepted flags and actual guided/row-path call counts. A
requested flag with no exercised path cannot count as an improvement.

`smol-profile` returns per-operation call counts and CPU wall time, including
barrier waits. Fused work is attributed to its leading operation. Output-head
matrix time is tagged by `result_output`. These instrumented request aggregates
include prefill and decode and are diagnostic, not isolated kernel timers or
unmodified performance measurements.

Improved requires three valid 64-token capped runs, all functional checks passing,
actual candidate execution, output identical to the matched-build control, and
at least five percent higher median decode rate. Early EOS is preserved and is
not accepted as an equal-length timing sample. There is no long-response gate in
this protocol and no inference about long-response stability. An old-build
baseline is preparation only and is never substituted for the matched-build control.

```bash
python3 scripts/xbox-smol-short.py --xbox <private-IP> --output <directory> --build-info <artifact-build-info.json>
```

Each profile saves its own result, graph, HTML report and full capped responses.
The summary preserves every run. Reports do not modify the earlier campaign.
The patch-specific correctness gate is `scripts/test-one-x-cpu-patch.sh`. It
builds an isolated copy of the pinned ggml with the optional patch and tests
Ministral-width Q4_K/Q6_K matrices, odd row tails, single-row operations,
multiple thread counts and stock fallback. It does not edit the submodule.
