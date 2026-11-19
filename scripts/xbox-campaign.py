#!/usr/bin/env python3
# Copyright (c) 2026 xllama contributors
# SPDX-License-Identifier: MIT
"""Serial, uncapped Ministral experiments with raw evidence and per-test reports."""

import argparse
import ast
import base64
import csv
import html
import hashlib
import ipaddress
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL = "ministral3-3b-instruct"
SHA256 = "9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8"
PROFILES = {
    "baseline": ("CPU reference", "8192 context, 6 threads, batch 512, microbatch 128, F16 KV, no lookup, CPU weights."),
    "gpu-1": ("Offload one layer", "Set SessionParams.n_gpu_layers=1; use SM5.1 signed-byte dot shaders after CPU-reference selftests."),
    "gpu-8": ("Offload eight layers", "Set only n_gpu_layers=8. Norms, attention and sampling remain CPU operations."),
    "gpu-27": ("Request all layer placements", "Set only n_gpu_layers=27 (26 repeating layers plus output). Supported matrix operations use D3D12."),
    "threads-4": ("Use four CPU threads", "Set only n_threads=4 for prefill and decode."),
    "threads-2": ("Use two CPU threads", "Set only n_threads=2 for prefill and decode."),
    "ubatch-64": ("Smaller prefill microbatch", "Set only n_ubatch=64; logical batch stays 512."),
    "ubatch-256": ("Larger prefill microbatch", "Set only n_ubatch=256; logical batch stays 512."),
    "kv-q8": ("Quantize the KV cache", "Set only kv_q8=true. The runtime also forces flash attention, which quantized V requires. Report actual fallback state."),
    "prompt-lookup": ("Enable prompt lookup", "Set only prompt_lookup=true; use the existing draft-free speculative decoder and count drafted/accepted tokens."),
}
HEADINGS = ["Choosing containers", "Drainage", "Growing medium", "Sunlight", "Choosing crops",
            "Planting", "Watering", "Feeding", "Pest prevention", "Weather protection", "Harvesting", "Seasonal cleanup"]
LONG_PROMPT = ("Write a complete practical field guide for a beginner growing vegetables in containers. "
               "Use all twelve numbered headings listed below, in order. Under EACH heading write "
               "one substantial paragraph of 80 to 100 words. Include practical reasons and concrete examples. "
               "Do not shorten the last sections or omit any heading. Aim for at least 1100 words overall. "
               "End with the exact line END OF GUIDE.\n" +
               "\n".join(f"{i}. {name}" for i, name in enumerate(HEADINGS, 1)))
STRESS_PROMPT = ("Write a complete 24-section practical manual about maintaining a small home garden. "
                 "Number every section from 1 through 24. Write 90 to 110 words in EACH section, "
                 "with concrete examples, varied advice and coherent transitions. Cover planning, light, "
                 "containers, beds, soil, compost, seeds, transplanting, watering, feeding, weeds, insects, "
                 "diseases, support, pruning, pollination, weather, succession, harvesting, storage, hygiene, "
                 "winter, budgeting and record keeping. Do not end early. End with END OF MANUAL.")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def quality_checks(text, tokens):
    words = text.split()
    lowered = text.lower()
    covered = sum(heading.lower() in lowered for heading in HEADINGS)
    grams = [tuple(words[i:i + 6]) for i in range(max(0, len(words) - 5))]
    repeated = 1 - len(set(grams)) / len(grams) if grams else 0
    return {"at_least_1024_output_tokens": tokens >= 1024,
            "at_least_700_words": len(words) >= 700,
            "all_twelve_headings": covered == 12,
            "final_marker": text.rstrip().endswith("END OF GUIDE"),
            "limited_six_word_repetition": repeated < 0.35,
            "word_count": len(words), "headings_present": covered,
            "repeated_six_gram_fraction": repeated}


def quality_passed(checks):
    return all(value for value in checks.values() if isinstance(value, bool))


def stress_checks(record):
    text = record.get("text", "")
    numbered = {int(value) for value in re.findall(r"(?m)^\s*(?:#{1,4}\s*)?(?:\*\*)?(\d{1,2})[.)]\s", text)}
    return {"at_least_2048_output_tokens": record.get("usage", {}).get("completion_tokens", 0) >= 2048,
            "all_24_sections": set(range(1, 25)) <= numbered,
            "final_marker": text.rstrip().endswith("END OF MANUAL"),
            "natural_stop": record.get("finish_reason") == "stop",
            "no_execution_error": not record.get("error"), "word_count": len(text.split())}


def aggregate(runs):
    """Independent medians are comparison statistics, never a fabricated raw run."""
    measured = [run for run in runs if not run.get("error") and run.get("xbox_metrics")]
    if len(measured) != len(runs):
        return {"error": "one or more repetitions failed", "xbox_metrics": {}}
    metrics = {}
    keys = set.intersection(*(set(run["xbox_metrics"]) for run in measured)) if measured else set()
    for key in keys:
        values = [run["xbox_metrics"][key] for run in measured]
        if all(isinstance(value, bool) for value in values):
            metrics[key] = all(values)
        elif all(isinstance(value, (int, float)) for value in values):
            metrics[key] = statistics.median(values)
    rates = [run["xbox_metrics"].get("decode_tokens", 0) / max(run["xbox_metrics"].get("decode_ms", 0) / 1000, 1e-12) for run in measured]
    # Preserve the median of rates, rather than a ratio of unrelated medians.
    if rates and min(rates) > 0:
        metrics["decode_ms"] = metrics["decode_tokens"] / statistics.median(rates) * 1000
    return {"error": None, "xbox_metrics": metrics}


def verdict(profile, baseline, candidate, quality_ok):
    if profile == "baseline":
        return False, "reference"
    if candidate.get("error") or not quality_ok:
        return False, "failed or quality gate not met"
    if baseline.get("error"):
        return False, "reference failed; no improvement conclusion"
    a, b = baseline.get("xbox_metrics", {}), candidate.get("xbox_metrics", {})
    if profile.startswith("gpu-") and not (b.get("configured_gpu_layers", 0) > 0 and b.get("gpu_dispatches", 0) > 0):
        return False, "GPU execution not demonstrated"
    if profile == "kv-q8" and not b.get("active_kv_q8"):
        return False, "Q8 cache fell back; requested change not applied"
    rate_a = a.get("decode_tokens", 0) / max(a.get("decode_ms", 0) / 1000, 1e-12)
    rate_b = b.get("decode_tokens", 0) / max(b.get("decode_ms", 0) / 1000, 1e-12)
    if rate_a <= 0 or a.get("decode_ms", 0) <= 0 or b.get("decode_ms", 0) <= 0:
        return False, "missing engine timing"
    if rate_b >= rate_a * 1.05:
        return True, "at least 5% faster; exploratory until repeated"
    memory_a, memory_b = a.get("sampled_peak_working_set_mib", 0), b.get("sampled_peak_working_set_mib", 0)
    if profile.startswith("ubatch-") and rate_b >= rate_a * 0.95:
        if a.get("prefill_tokens") == b.get("prefill_tokens") and a.get("prefill_ms", 0) > 0 and 0 < b.get("prefill_ms", 0) <= a["prefill_ms"] * 0.90:
            return True, "at least 10% faster prefill; decode within 5%; exploratory"
        if profile == "ubatch-64" and memory_a > 0 and 0 < memory_b <= memory_a * 0.90:
            return True, "at least 10% less observed RSS; decode within 5%; exploratory"
    if profile == "kv-q8" and memory_a > 0 and 0 < memory_b <= memory_a * 0.90 and rate_b >= rate_a * 0.95:
        return True, "at least 10% less observed RSS; speed within 5%; exploratory"
    return False, "improvement threshold not reached"


class Campaign:
    def __init__(self, xbox, output, build_info=None):
        self.base = f"http://{xbox}:11434"
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.results = []
        self.build_info = build_info

    def run(self, profile, messages, label):
        directory = self.output / profile
        directory.mkdir(exist_ok=True)
        path = directory / f"{label}.json"
        payload = {"model": MODEL, "xbox_experiment": profile, "xbox_fresh": True,
                   "messages": messages, "temperature": 0, "seed": 42, "stream": True}
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("done") and not record.get("error") and record.get("request") == payload and record.get("build_info") == self.build_info:
                return record
            previous = f"{label}.previous-{time.time_ns()}"
            path.rename(directory / f"{previous}.json")
            raw_previous = directory / f"{label}.sse.txt"
            if raw_previous.exists():
                raw_previous.rename(directory / f"{previous}.sse.txt")
        record = {"profile": profile, "label": label, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "request": payload, "text": "", "done": False, "error": None,
                  "first_text_seconds": None, "usage": {}, "progress": [], "build_info": self.build_info}
        print(f"START {profile}/{label} (reply budget omitted; EOS/context)", flush=True)
        begin = time.perf_counter()
        last_notice = begin
        try:
            for attempt in range(8):
                request = urllib.request.Request(self.base + "/v1/chat/completions",
                    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                try:
                    response = self.opener.open(request, timeout=180)
                    break
                except urllib.error.HTTPError as error:
                    body = error.read().decode()
                    if error.code != 503 or "busy" not in body or attempt == 7:
                        raise RuntimeError(f"HTTP {error.code}: {body}") from error
                    print(f"WAIT {profile}/{label}: Xbox busy", flush=True)
                    time.sleep(5)
            with response, (directory / f"{label}.sse.txt").open("w") as raw_stream:
                for raw in response:
                    raw_stream.write(raw.decode("utf-8"))
                    now = time.perf_counter()
                    if now - begin > 45 * 60:
                        raise TimeoutError("45-minute request watchdog; response aborted")
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        record["done"] = True
                        break
                    event = json.loads(body)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    if event.get("usage"):
                        record["usage"] = event["usage"]
                    if event.get("xbox_metrics"):
                        record["xbox_metrics"] = event["xbox_metrics"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {}).get("content") or ""
                        if delta:
                            if record["first_text_seconds"] is None:
                                record["first_text_seconds"] = now - begin
                            record["text"] += delta
                            record["progress"].append([round(now - begin, 4), len(record["text"])])
                        if choice.get("finish_reason"):
                            record["finish_reason"] = choice["finish_reason"]
                    if now - last_notice > 30:
                        print(f"PROGRESS {profile}/{label}: {now - begin:.0f}s; {len(record['text'])} characters received", flush=True)
                        dump(self.output / "progress.json", {"profile": profile, "label": label,
                             "elapsed_seconds": now - begin, "characters": len(record["text"])})
                        last_notice = now
            if not record["done"] or "xbox_metrics" not in record:
                raise RuntimeError("Incomplete stream or experiment telemetry absent; package may be old")
            if record["xbox_metrics"].get("context") != 8192:
                raise RuntimeError("Effective context differs from protocol")
        except Exception as error:
            record["error"] = str(error)
        finally:
            record["total_seconds"] = time.perf_counter() - begin
            dump(path, record)
        m = record.get("xbox_metrics", {})
        rate = m.get("decode_tokens", 0) / max(m.get("decode_ms", 0) / 1000, 1e-12)
        print(f"END {profile}/{label}: tokens={m.get('decode_tokens')} decode={rate:.3f}/s finish={record.get('finish_reason')} error={record['error']}", flush=True)
        return record

    def test(self, profile):
        checks = []
        tests = [("exact", "Reply with exactly: ORCHID", "ORCHID"),
                 ("arithmetic", "What is 19 multiplied by 23? Reply with only the integer.", "437"),
                 ("extraction", "Record: apple=red; pear=green; lemon=yellow. Reply with only the color of pear.", "green")]
        for label, prompt, expected in tests:
            result = self.run(profile, [{"role": "user", "content": prompt}], label)
            checks.append({"test": label, "passed": result["text"].strip().rstrip(".").lower() == expected.lower() and not result["error"],
                           "actual": result["text"], "expected": expected, "error": result["error"]})
            if result["error"]:
                break
        if not any(item.get("error") for item in checks):
            structured = self.run(profile, [{"role":"user", "content":"Return only valid JSON with exactly these fields: status is the string ok; count is the integer 3."}], "structured")
            try:
                correct_json = json.loads(structured["text"]) == {"status":"ok", "count":3}
            except (ValueError, TypeError):
                correct_json = False
            checks.append({"test":"structured", "passed":correct_json and not structured["error"], "actual":structured["text"], "error":structured["error"]})
            code = self.run(profile, [{"role":"user", "content":"Write only a Python function named square with one argument x that returns x*x. No explanation."}], "code")
            try:
                source = code["text"].strip()
                if source.startswith("```") and source.endswith("```"):
                    source = "\n".join(source.splitlines()[1:-1])
                tree = ast.parse(source)
                function = tree.body[0]
                expression = function.body[0].value
                correct_code = (len(tree.body)==1 and isinstance(function,ast.FunctionDef) and function.name=="square" and
                    len(function.args.args)==1 and function.args.args[0].arg=="x" and len(function.body)==1 and
                    isinstance(function.body[0],ast.Return) and isinstance(expression,ast.BinOp) and
                    isinstance(expression.op,ast.Mult) and isinstance(expression.left,ast.Name) and expression.left.id=="x" and
                    isinstance(expression.right,ast.Name) and expression.right.id=="x")
            except (SyntaxError, AttributeError, IndexError):
                correct_code = False
            checks.append({"test":"code", "passed":correct_code and not code["error"], "actual":code["text"], "error":code["error"]})
        if not any(item.get("error") for item in checks):
            recall = self.run(profile, [{"role": "user", "content": "Remember my code word is ORCHID. Reply OK."},
                                      {"role": "assistant", "content": "OK"},
                                      {"role": "user", "content": "What is my code word? Reply with one word."}], "recall")
            checks.append({"test": "recall", "passed": recall["text"].strip().rstrip(".") == "ORCHID" and not recall["error"], "actual": recall["text"]})
        # Preserve quality failures but still measure a long answer unless execution failed.
        if any(item.get("error") for item in checks):
            long_run = {"error": "short execution gate failed", "usage": {}, "text": ""}
        else:
            long_run = self.run(profile, [{"role": "user", "content": LONG_PROMPT}], "long-01")
        long_checks = quality_checks(long_run.get("text", ""), long_run.get("usage", {}).get("completion_tokens", 0))
        good = len(checks) == 6 and all(item["passed"] for item in checks) and quality_passed(long_checks) and long_run.get("finish_reason") == "stop"
        reference = self.results[0]["long_run"] if self.results else long_run
        reference_good = self.results[0]["quality_passed"] if self.results else good
        improved, status = verdict(profile, reference, long_run, good and reference_good)
        result = {"profile": profile, "changed": PROFILES[profile][0], "implementation": PROFILES[profile][1],
                  "quality_checks": checks, "long_checks": long_checks, "quality_passed": good,
                  "long_run": long_run, "improved": improved, "status": status, "replicates": 1}
        if not long_run.get("error") and long_run.get("usage", {}).get("completion_tokens", 0) < 1024:
            followup = self.run(profile, [{"role":"user", "content":STRESS_PROMPT}], "length-followup")
            result["length_followup"] = followup
            result["length_followup_checks"] = stress_checks(followup)
        self.results.append(result)
        dump(self.output / profile / "result.json", result)
        render(self.output, self.results)
        return result

    def confirm_and_stress(self):
        passing = [result for result in self.results[1:] if result["quality_passed"]]
        def rate(result):
            m = result["long_run"].get("xbox_metrics", {})
            return m.get("decode_tokens", 0) / max(m.get("decode_ms", 0) / 1000, 1e-12)
        # Recheck the fastest usable candidate, even if no initial improvement won.
        selected = sorted(passing, key=rate, reverse=True)[:1]
        memory_winners = [result for result in passing if result["improved"] and result not in selected and result["profile"] in ("kv-q8", "ubatch-64")]
        if memory_winners:
            selected.append(min(memory_winners, key=lambda result: result["long_run"]["xbox_metrics"]["sampled_peak_working_set_mib"]))
        targets = [self.results[0]] + selected
        for result in targets:
            result["repetition_runs"] = [result["long_run"]]
        # Alternate the reference and candidates to expose drift over the campaign.
        for repetition in (2, 3):
            for result in targets:
                record = self.run(result["profile"], [{"role": "user", "content": LONG_PROMPT}], f"long-{repetition:02}")
                result["repetition_runs"].append(record)
                result["replicates"] = len(result["repetition_runs"])
                result["comparison_statistics"] = aggregate(result["repetition_runs"])
                result["repeat_quality_passed"] = all(
                    not run.get("error") and run.get("finish_reason") == "stop" and
                    quality_passed(quality_checks(run.get("text", ""), run.get("usage", {}).get("completion_tokens", 0)))
                    for run in result["repetition_runs"])
                for item in self.results:
                    comparison = item.get("comparison_statistics", item["long_run"])
                    reference = self.results[0].get("comparison_statistics", self.results[0]["long_run"])
                    good = item["quality_passed"] and item.get("repeat_quality_passed", True) and self.results[0]["quality_passed"] and self.results[0].get("repeat_quality_passed", True)
                    item["improved"], item["status"] = verdict(item["profile"], reference, comparison, good)
                    if item.get("replicates", 1) >= 3:
                        item["status"] = item["status"].replace("exploratory until repeated", "median of three runs; small sample").replace("exploratory", "median of three runs; small sample")
                    dump(self.output / item["profile"] / "result.json", item)
                render(self.output, self.results)
        for result in targets:
            record = self.run(result["profile"], [{"role": "user", "content": STRESS_PROMPT}], "stress-01")
            result["stress_run"] = record
            result["stress_checks"] = stress_checks(record)
            if not quality_passed(result["stress_checks"]):
                result["improved"] = False
                result["status"] += "; extended stress gate failed"
            dump(self.output / result["profile"] / "result.json", result)
            render(self.output, self.results)
        if not quality_passed(self.results[0].get("stress_checks", {})):
            for result in self.results[1:]:
                result["improved"] = False
                result["status"] += "; reference extended stress gate failed"
                dump(self.output / result["profile"] / "result.json", result)
            render(self.output, self.results)


def render(output, results):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/xllama-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    baseline = results[0]
    def metrics(item):
        run = item["long_run"]
        m = item.get("comparison_statistics", run).get("xbox_metrics", {})
        rate = m.get("decode_tokens", 0) / (m["decode_ms"] / 1000) if m.get("decode_ms", 0) > 0 else None
        return [rate, m.get("sampled_peak_working_set_mib"), m.get("decode_tokens"),
                m.get("prefill_ms"), run.get("first_text_seconds"), m.get("gpu_dispatches", 0)]
    for result in results:
        profile = result["profile"]
        directory = output / profile
        directory.mkdir(exist_ok=True)
        a, b = metrics(baseline), metrics(result)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for axis, index, title in [(axes[0], 0, "Engine decode (tokens/s; higher is better)"),
                                    (axes[1], 1, "Observed peak RSS (MiB; lower is better)")]:
            values = [a[index], b[index]]
            positions = [i for i, value in enumerate(values) if value is not None]
            bars = axis.bar(positions, [values[i] for i in positions], color=["#6d7782" if i == 0 else "#2675b9" for i in positions])
            axis.set_xticks([0,1], ["CPU reference", profile])
            axis.set_xlim(-0.6, 1.6)
            axis.set_title(title, fontsize=10)
            axis.bar_label(bars, fmt="%.2f", padding=3)
            axis.set_ylim(0, max([value for value in values if value is not None] + [1]) * 1.18)
            for i, value in enumerate(values):
                if value is None:
                    axis.text(i, 0.05, "Not measured", ha="center", transform=axis.get_xaxis_transform())
        for item, label, color in [(baseline, "CPU reference", "#6d7782"), (result, profile, "#2675b9")]:
            points = item["long_run"].get("progress", [])
            if points:
                axes[2].plot([x[0] for x in points], [x[1] for x in points], label=label, color=color)
        axes[2].set(xlabel="Request elapsed (seconds)", ylabel="Received characters", title="Long-response progress")
        if axes[2].lines:
            axes[2].legend(fontsize=8)
        count = result["replicates"]
        description = "exploratory single run" if count == 1 else f"median of {count} runs; small sample"
        fig.suptitle(f"{profile} | Improved: {str(result['improved']).lower()} | {description}")
        image_path = directory / "comparison.png"
        fig.savefig(image_path, dpi=160)
        fig.savefig(directory / "comparison.svg")
        plt.close(fig)
        fields = {"Changed": result["changed"], "How implemented": result["implementation"],
                  "Improved": str(result["improved"]).lower(), "Status": result["status"],
                  "Output tokens": b[2], "Finish reason": result["long_run"].get("finish_reason"),
                  "Engine decode tokens/s": round(b[0], 3) if b[0] is not None else "Not measured", "Prefill ms": b[3],
                  "First received text seconds": b[4], "Observed peak RSS MiB": b[1],
                  "GPU dispatches": b[5], "Quality gates passed": result["quality_passed"],
                  "Error": result["long_run"].get("error"), "Repeat count": count,
                  "Statistics": "Independent medians for throughput, RSS, tokens and prefill when repeated. First-text and progress curve show the initial run.",
                  "Initial total request seconds": result["long_run"].get("total_seconds"), "Initial engine decode ms": result["long_run"].get("xbox_metrics",{}).get("decode_ms"), "Initial engine settings": result["long_run"].get("xbox_metrics",{})}
        body = "<table>" + "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in fields.items()) + "</table>"
        image_data = base64.b64encode(image_path.read_bytes()).decode()
        checks = html.escape(json.dumps({"short": result["quality_checks"], "long": result["long_checks"]}, indent=2))
        answer = html.escape(result["long_run"].get("text", ""))
        extra = ""
        if result.get("repetition_runs"):
            extra += "<h2>Individual repetitions</h2><p>Each raw JSON and SSE file is retained alongside this report.</p>"
            for run in result["repetition_runs"]:
                m = run.get("xbox_metrics", {})
                repeat = {"label": run.get("label"), "tokens": m.get("decode_tokens"), "decode_ms": m.get("decode_ms"), "peak_rss_mib": m.get("sampled_peak_working_set_mib"), "finish": run.get("finish_reason"), "error": run.get("error")}
                extra += f"<pre>{html.escape(json.dumps(repeat, indent=2))}</pre>"
        if result.get("stress_run"):
            stress = result["stress_run"]
            extra += "<h2>Extended response stress test</h2><pre>" + html.escape(json.dumps({"checks":result["stress_checks"], "metrics":stress.get("xbox_metrics", {}), "error":stress.get("error")}, indent=2)) + "</pre><pre>" + html.escape(stress.get("text", "")) + "</pre>"
        if result.get("length_followup"):
            followup = result["length_followup"]
            extra += "<h2>Length follow-up, separate workload</h2><p>The primary response missed 1024 output tokens. This follow-up preserves that initial failure and does not replace its comparison.</p><pre>" + html.escape(json.dumps({"checks":result["length_followup_checks"], "metrics":followup.get("xbox_metrics", {}), "error":followup.get("error")}, indent=2)) + "</pre><pre>" + html.escape(followup.get("text", "")) + "</pre>"
        document = f"<!doctype html><meta charset='utf-8'><title>{profile} experiment</title><style>body{{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #ccc}}th{{width:28%}}img{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><h1>{profile}</h1><p>Xbox One X, Ministral 3 3B Q4_K_M. Context 8192. Reply budget omitted. Synthetic workload. One run is exploratory. Repeated bars are independent medians, not a single raw request.</p>{body}<img alt='Throughput, memory and initial output progress comparison' src='data:image/png;base64,{image_data}'><h2>Quality checks</h2><pre>{checks}</pre><h2>Initial full response</h2><pre>{answer}</pre>{extra}"
        (directory / "report.html").write_text(document)
        rows.append({"profile": profile, "decode_tokens_s": round(b[0], 3) if b[0] is not None else None, "output_tokens": b[2],
                     "peak_rss_mib": b[1], "gpu_dispatches": b[5], "quality_passed": result["quality_passed"],
                     "improved": result["improved"], "status": result["status"]})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for axis, key, title in [(axes[0], "decode_tokens_s", "Engine decode throughput (tokens/s)"),
                              (axes[1], "peak_rss_mib", "Observed peak working set (MiB)")]:
        positions = [i for i, row in enumerate(rows) if row[key] is not None]
        bars = axis.bar(positions, [rows[i][key] for i in positions], color=["#24884f" if rows[i]["improved"] else "#6d7782" for i in positions])
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        axis.set_xticks(range(len(rows)), [row["profile"] for row in rows], rotation=20, ha="right")
        axis.set_xlim(-0.6, len(rows) - 0.4)
        axis.set_title(title)
        axis.set_ylim(0, max([row[key] for row in rows if row[key] is not None] + [1]) * 1.2)
        for i, row in enumerate(rows):
            if row[key] is None:
                axis.text(i, 0.03, "Not measured", rotation=90, ha="center", transform=axis.get_xaxis_transform())
    fig.suptitle("Ministral 3 3B on Xbox One X | green = improvement demonstrated under protocol")
    fig.savefig(output / "overview.png", dpi=160)
    fig.savefig(output / "overview.svg")
    plt.close(fig)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    links = "".join(f"<li><a href='{row['profile']}/report.html'>{row['profile']}: Improved {str(row['improved']).lower()} — {html.escape(row['status'])}</a></li>" for row in rows)
    (output / "index.html").write_text(f"<!doctype html><meta charset='utf-8'><title>Ministral One X experiments</title><h1>Ministral One X experiments</h1><p>Serial, one-variable tests. Open each report for its table, graph, full response and checks. False includes improvement not demonstrated.</p><img style='max-width:100%' alt='All measured profiles throughput and memory' src='overview.png'><ul>{links}</ul>")
    dump(output / "summary.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbox", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-info", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--initial-only", action="store_true", help="Stop before confirmation and extended stress workloads")
    args = parser.parse_args()
    if not ipaddress.IPv4Address(args.xbox).is_private:
        parser.error("Use the Xbox's private LAN address")
    build_info = json.loads(args.build_info.read_text())
    campaign = Campaign(args.xbox, args.output, build_info)
    dump(args.output / "method.json", {"model": MODEL, "sha256": SHA256, "hardware": "Xbox One X (operator-declared)",
         "profiles": PROFILES, "reply_budget": "omitted; EOS/context", "context": 8192,
         "protocol": "docs/ministral-one-x-campaign.md", "primary_minimum_tokens": 1024,
         "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "build_info": build_info})
    profiles = ["baseline"] + [x for x in args.profiles if x != "baseline"]
    for profile in profiles:
        campaign.test(profile)
    if not args.initial_only:
        campaign.confirm_and_stress()
    print("CAMPAIGN COMPLETE; manual response review still required", flush=True)


if __name__ == "__main__":
    main()
