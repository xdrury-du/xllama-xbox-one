# Xbox One X CPU experiment

This fork's initial target is an Xbox One X in Developer Mode, deployed from
a Mac through Xbox Device Portal. A successful Windows build establishes
packaging only. No One X model load, output correctness, speed or memory
claim is made until the console test below passes.

## Build

The [Xbox One CPU workflow](../.github/workflows/build-xbox-one.yml) uses a
GitHub-hosted Windows runner. A Windows PC or VM on the Mac is unnecessary.
The `xllama-xbox-one-cpu` artifact contains the MSIX, test certificate and
available APPX dependencies. Download artifacts only from a successful run
for the exact experiment commit.

To reproduce on Windows with the upstream prerequisites:

```powershell
./scripts/build-uwp.ps1 -Configuration Release -Platform x64 -Backend llamacpp -XboxOne
```

`-XboxOne` propagates `/arch:AVX` and an ISA guard to the app, ggml and
llama-common builds. Jaguar supports AVX, but lacks AVX2, FMA3 and BMI2.
The profile disables xllama's GGUF D3D12 backend instead of running its
Series-console shaders. GPU acceleration remains a separate experiment.
ORT text inference is excluded. Some upstream ONNX runtime DLLs remain
package dependencies for other compiled app surfaces; those surfaces have
not been validated on One X.

## Model policy

The profile admits Google's Gemma 3 270M baseline and Mistral AI's Ministral
3 3B Instruct 2512 text-only candidate. Source, filename and SHA-256 are pinned
in [`xbox_one_profile.h`](../include/xllama/xbox_one_profile.h) and the
[experiment catalogue](../uwp/models/manifest.xbox-one.json). Gemma remains the
first-launch default. See the [Ministral integration review](ministral3-one-x.md)
for the artifact, prompt format and validation scope. The Mac client is now
versioned as [`scripts/xbox-chat.py`](../scripts/xbox-chat.py).

The restriction concerns the model developer and disclosed base/teacher
lineage, not contributors' ancestry. Undisclosed training details cannot be
certified. No other model is cleared implicitly because its publisher has a
US or Western European address.

The One X package contains only the experiment catalogue. It ignores
LocalState catalogue overrides, resets unapproved saved model selections,
rejects other model ids at the session and inference entry points, disables
automatic USB provisioning and rejects unapproved downloader requests.
A missing experiment catalogue returns an empty list rather than enabling
the upstream fallback. These controls prevent normal app flows from
downloading or selecting excluded models. They do not authenticate arbitrary
files manually substituted through Developer Mode under the approved name.

Upstream catalogues, tests and historical benchmark records remain in source
for comparison. They are not additional models offered by this package.

## Install and first console test

1. Keep the Xbox at Dev Home and open its `https://<ip>:11443` Device Portal.
2. Install the artifact using the [package installation guide](install-release.md).
   Use the experiment artifact instead of an upstream Series-console release.
   Browser certificate installation varies; the guide includes a CLI fallback.
3. In Dev Home, open the installed tile's details and set **App type** to **Game**.
4. Launch xllama. Confirm the model id is `gemma3-270m`. Its download is about
   253 MB. Start with one short prompt, such as `Write one sentence about rain.`
5. Confirm readable output, normal stopping and a second successful turn.
6. Return to Dev Home and copy `LocalState/xllama.log` from Device Portal's
   file explorer. Record the source commit, MSIX version, model digest,
   console model/OS, available memory, load time and generation measurements.

The catalogue requests a 1024-token context and a 128-token generation limit.
Check the actual log before comparing results. Do not run the upstream
benchmark script with a Series S/X hardware label: its accepted hardware
choices have not yet been extended for One X. Keep One X receipts separate.

If the app closes before logging, collect its crash dump. An illegal
instruction points to a remaining ISA or runtime dependency issue; an
installation failure alone is not evidence that the CPU port failed.

## Next experiments

After installing the Ministral update, select and download the model through
the Mac client, validate two text turns, then switch back to Gemma. Compare fixed prompts and generation
lengths while varying one of thread count, context size, quantization or
kernel implementation. Report both quality and speed. The One X is not a
Series X, so Series-console throughput cannot be transferred to this device.
