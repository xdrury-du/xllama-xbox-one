# Ministral 3 3B Instruct on Xbox One X

This change adds the text-only path for Mistral AI's Ministral 3 3B Instruct
2512 to the existing One X CPU experiment. Gemma 3 270M remains the first-launch
default and the smaller comparison model. The app identity stays unchanged so
an in-place update preserves downloaded weights, settings and chat history.

## What changed and why

### Reuse the existing inference engine

The pinned llama.cpp revision `d81235049384534c167caea52b85a694f6103d14`
already implements the GGUF's `mistral3` architecture. The integration does
not require a new backend, dependency update, weight conversion or changes
inside the submodule. The CPU AVX build and disabled GGUF D3D12 backend remain
the One X execution profile.

This keeps a model integration separate from a runtime upgrade. If a failure
appears, we can investigate the new artifact, prompt format or resource use
without also debugging an unrelated engine change.

### Admit one exact official artifact

[`xbox_one_profile.h`](../include/xllama/xbox_one_profile.h) now holds a table
of two reviewed model pins. An id must match a table entry. A download must
match one entry's complete source, filename and digest together. A source
from one model combined with the other's filename or hash is rejected.

The [experiment catalogue](../uwp/models/manifest.xbox-one.json) adds:

| Field               | Value                                                              |
| ------------------- | ------------------------------------------------------------------ |
| Id                  | `ministral3-3b-instruct`                                           |
| Source              | `mistralai/Ministral-3-3B-Instruct-2512-GGUF`                      |
| Revision            | `eb599d408350ea2bb60452cb86be7c7b2fc28227`                         |
| File                | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`                         |
| Bytes               | `2147023008`                                                       |
| SHA-256             | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Context             | 1024 tokens                                                        |
| Initial reply limit | 128 tokens                                                         |
| GPU layers          | 0                                                                  |

The source revision fixes which file we request, while the SHA-256 verifies
the actual downloaded bytes. The model stays outside the MSIX and downloads
directly from its official publisher. The catalogue and C++ policy are checked
against one another by a host test.

The artifact is approximately 2.15 GB on disk. That is not a peak-RAM estimate.
We keep the same 1024-token context as the Gemma baseline rather than selecting
the model's advertised 256K context. Weight loading, KV cache and computation
buffers all consume the Xbox process's memory budget. SessionHub unloads the
previous model before loading the selected one.

### Render the format Ministral was trained to read

Plain chat previously selected Gemma, Llama, Phi or ChatML formatting. A
Ministral id would fall through to ChatML. The engine could load the weights
while receiving the wrong conversation markers.

[`chat_prompt.cpp`](../src/bridge/chat_prompt.cpp) adds a `Ministral3` format
through the existing `ChatFormat` interface:

```text
[SYSTEM_PROMPT]Be concise.[/SYSTEM_PROMPT]
[INST]First user message[/INST]First assistant answer</s>
[INST]Next user message[/INST]
```

The line breaks above are illustrative. The renderer does not insert them.
The user turn closes with `[/INST]`, but the assistant turn closes with
`</s>`. There is no extra assistant header before generation. This asymmetry
requires a small format-specific renderer instead of filling in the symmetric
role fields used by ChatML.

The pinned GGUF declares `add_bos_token=true`, so the existing tokenizer
inserts `<s>`. Emitting another literal BOS in the renderer would duplicate
it. The EOS stop, system-prefix calculation and continuation rendering are
also covered. Both the native UI and LAN API use this shared implementation,
including the prompt-budget calculation.

This implements the plain-text subset of the template embedded in the
reviewed GGUF. It does not add images, tools or the vendor's Le Chat default
system prompt. xllama continues to supply its own explicit system prompt.
An artifact with a different template needs another review.

### Keep the experiment text only

The catalogue includes only the language-model GGUF. It excludes the separate
841,501,856-byte `mmproj` file. No vision encoder is initialized, and the Mac
client accepts text strings rather than image content. The CPU text path can
therefore be tested without adding a multimodal pipeline or its memory cost.

### Select and download from the Mac

[`scripts/xbox-chat.py`](../scripts/xbox-chat.py) versions the existing local
browser client and adds a model selector plus a download/load button. It
discovers installed models through `/v1/models` and streams `/api/pull`
progress. Selecting a different model starts a fresh browser conversation so
history from the other model is not silently reused in comparisons.

The browser talks to a loopback-only Python server on the Mac. That server
forwards only the approved model ids to the Xbox, retains the local request
token check, and serializes download/generation requests. It records chat
timings in `runs.jsonl` and pull outcomes in `pulls.jsonl`. Both are local
experiment data and are excluded from Git.

```bash
python3 scripts/xbox-chat.py --xbox <console-ip>
```

The Xbox still performs inference and must keep xllama in the foreground.
The LAN API remains unauthenticated and intended for a trusted home network.

## Provenance

Mistral AI is the French publisher of this official model and quantization.
Its [technical report](https://arxiv.org/html/2601.08584v1) identifies Mistral
Small 3.1 as the pretraining parent/teacher and Mistral Medium as the instruct
distillation teacher. No excluded-origin parent or teacher is identified in
that disclosed recipe. This is a review of disclosed lineage, not a claim
that every unpublished training detail has been independently audited.

The [official artifact](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF)
is Apache-2.0 licensed. No model weights are committed to this repository.

## Verification and limitations

- The downloaded file's SHA-256 matched the official Hugging Face LFS digest.
- The pinned runtime loaded it on the Mac with CPU execution, GPU support
  disabled, ordinary buffered loading, 1024 context tokens and six workers.
  It completed `Reply with exactly: Ministral is running.` correctly and
  recalled `ORCHID` from a preceding conversation turn. This is
  a host compatibility check and provides no One X throughput or RAM result.
- Four renderer cases matched the embedded GGUF Jinja template byte for byte
  after excluding its tokenizer-added BOS: single turn, history, empty system
  and Unicode. Focused C++ tests also cover stopping and continuation.
- Policy tests reject mixed pins and the vision projector, and check that the
  packaged catalogue agrees with the C++ table.
- Mac proxy tests cover model forwarding, discovery, text-only messages,
  local request authorization and incomplete download/generation streams.
- The Windows workflow checks the exact two-model package catalogue and
  rejects bundled weights or an extra unfiltered catalogue.

The updated package still requires console validation: install it in place,
confirm **App type: Game**, download/load Ministral, generate two turns and
switch back to Gemma. Record errors as outcomes. Do not describe the host
test or a successful MSIX build as a successful One X model run.

This change establishes compatibility and model selection. Quantization
comparisons, thread sweeps and kernel optimization are subsequent experiments.
