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
an outcome, not a completed answer. A 45-minute request watchdog may abort
a runaway response and must be recorded as an aborted test.

## Verdict and reports

Each report includes the exact settings and implementation change, raw
outputs, engine and request timing, output length, sampled RSS, actual GPU
activity, quality checks, errors, a comparison table and standalone graphs.
Plot measured decode throughput separately from prefill/first-text latency.
Output-progress curves use received characters, not an assumed one-token
per SSE chunk conversion.

`Improved = true` requires a successful long response, passing correctness
checks and at least 5 percent higher measured decode throughput than the
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
