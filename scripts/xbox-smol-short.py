#!/usr/bin/env python3
# Copyright (c) 2026 xllama contributors
# SPDX-License-Identifier: MIT
"""Capped, serial tests of explicitly implemented One X CPU techniques."""
import argparse
import importlib.util
import json
import statistics
from pathlib import Path

spec = importlib.util.spec_from_file_location('xbox_campaign', Path(__file__).with_name('xbox-campaign.py'))
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
PERFORMANCE_PROMPT = 'Explain how a bicycle works, including pedaling, gears, steering and braking. Give concrete mechanical details.'
CHECKS = [('exact', 'Reply with exactly: ORCHID', 'ORCHID', 8),
          ('arithmetic', 'What is 19 multiplied by 23? Reply with only the integer.', '437', 8),
          ('extraction', 'Record: apple=red; pear=green; lemon=yellow. Reply with only the color of pear.', 'green', 8),
          ('json', 'Return only valid JSON with exactly these fields: status is the string ok; count is the integer 3.', None, 32)]
FLAGS = {'baseline': 0, 'smol-base': 0, 'smol-guided': 2, 'smol-row': 4, 'smol-both': 6, 'smol-profile': 1}

def rate(run):
    m = run.get('xbox_metrics', {})
    return m.get('decode_tokens', 0) * 1000 / max(m.get('decode_ms', 0), 1e-9)

def bounded_record(run, cap):
    m = run.get('xbox_metrics', {})
    return (not run.get('error') and run.get('done') and
            run.get('request', {}).get('max_tokens') == cap and
            0 < m.get('decode_tokens', 0) <= cap and
            m.get('reply_token_budget') == cap and not m.get('auto_reply_budget', True))

def evaluate(profile, checks, runs, reference=None):
    valid = all(bounded_record(r, 64) and r['xbox_metrics']['decode_tokens'] == 64 and
                r.get('finish_reason') == 'length' for r in runs)
    applied = all(r.get('xbox_metrics', {}).get('cpu_experiment_flags', 0) == FLAGS[profile] for r in runs)
    if FLAGS[profile] & 2:
        applied &= all(r.get('xbox_metrics', {}).get('cpu_guided_calls', 0) > 0 for r in runs)
    if FLAGS[profile] & 4:
        applied &= all(r.get('xbox_metrics', {}).get('cpu_row_calls', 0) > 0 for r in runs)
    speed = statistics.median(rate(r) for r in runs) if valid else None
    same = None if reference is None else all(r['text'] == reference['runs'][0]['text'] for r in runs)
    improved = bool(reference and valid and applied and all(c['passed'] for c in checks) and same and
                    reference['valid_timing'] and speed >= reference['median_decode_tps'] * 1.05)
    return {'profile': profile, 'checks': checks, 'runs': runs, 'valid_timing': valid,
            'change_exercised': applied, 'same_text_as_reference': same, 'median_decode_tps': speed,
            'improved': improved, 'scope': 'short 64-token generation; no long-response conclusion'}

def render(root, results):
    # Keep the common plotting dependency out of transport and verdict unit tests.
    import base64
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    measured = [r for r in results if r['median_decode_tps'] is not None]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    if measured:
        bars = ax.bar([r['profile'] for r in measured], [r['median_decode_tps'] for r in measured],
                      color=['#24864f' if r['improved'] else '#627486' for r in measured])
        ax.bar_label(bars, fmt='%.3f', padding=4)
        ax.set_ylim(0, max(r['median_decode_tps'] for r in measured) * 1.2)
    ax.set(ylabel='Median engine decode tokens / second', title='Ministral short tests | 64-token cap, three repetitions')
    fig.savefig(root/'comparison.png', dpi=150); fig.savefig(root/'comparison.svg'); plt.close(fig)
    table = ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['profile'],r['median_decode_tps'],r['valid_timing'],r['change_exercised'],r['same_text_as_reference'],r['improved']])+'</tr>' for r in results)
    img = base64.b64encode((root/'comparison.png').read_bytes()).decode()
    doc = '<!doctype html><meta charset="utf-8"><title>Ministral short tests</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px}td,th{border:1px solid #ccc;padding:10px}table{border-collapse:collapse}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Ministral short tests</h1><p>Three measured repetitions per profile; each has an explicit 64-token cap. Quality checks use caps of 8 or 32. Instrumented profiling requests are separate diagnostics and excluded from speed comparisons. Improved requires valid equal-length timings, actual change-execution counters, passing functional checks, identical output and at least 5% faster median decode than the same-build control.</p><table><tr><th>Profile</th><th>Median tok/s</th><th>Valid timing</th><th>Change exercised</th><th>Same output</th><th>Improved</th></tr>'+table+'</table><img src="data:image/png;base64,'+img+'"><p>Individual request JSON, raw SSE streams and complete capped outputs are preserved in each profile directory. The baseline-only pre-update run is preparation, not a candidate comparison.</p>'
    for item in results:
        name = item['profile']
        reference = results[0]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        pair = [reference] if item is reference else [reference, item]
        for ax, title, vals in [
            (axes[0], 'Median decode tokens/s', [x['median_decode_tps'] for x in pair]),
            (axes[1], 'Median observed RSS MiB', [statistics.median(r['xbox_metrics']['sampled_peak_working_set_mib'] for r in x['runs']) for x in pair])]:
            if any(v is None for v in vals):
                ax.text(0.5, 0.5, 'Comparable measurement unavailable', ha='center', va='center', transform=ax.transAxes)
            else:
                bars = ax.bar([x['profile'] for x in pair], vals, color='#627486')
                ax.bar_label(bars, fmt='%.3f', padding=3)
                ax.set_ylim(0, max(vals + [1]) * 1.2)
            ax.set_title(title)
        fig.suptitle(name + ' | Improved: ' + str(item['improved']).lower())
        fig.savefig(root/name/'comparison.png', dpi=150)
        fig.savefig(root/name/'comparison.svg')
        plt.close(fig)
        png = base64.b64encode((root/name/'comparison.png').read_bytes()).decode()
        detail = {k:v for k,v in item.items() if k != 'runs'}
        metric_table='<table>'+''.join('<tr><th>'+html.escape(str(k))+'</th><td>'+html.escape(str(v))+'</td></tr>' for k,v in detail.items() if k!='checks')+'</table>'
        outputs = ''.join('<h2>Repetition '+str(i+1)+'</h2><pre>'+html.escape(r['text'])+'</pre><pre>'+html.escape(json.dumps(r['xbox_metrics'],indent=2))+'</pre>' for i,r in enumerate(item['runs']))
        (root/name/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(name)+'</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>'+html.escape(name)+'</h1><p>Every measured reply was capped at 64 tokens. Three repetitions, fresh KV, warm weights. Profiling diagnostics are excluded from speed verdicts.</p><img src="data:image/png;base64,'+png+'"><h2>Measurements and verdict</h2>'+metric_table+'<h2>Functional checks</h2><pre>'+html.escape(json.dumps(item['checks'],indent=2))+'</pre>'+outputs)
    (root/'report.html').write_text(doc)
    transport.dump(root/'results.json', results)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xbox', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--build-info', type=Path, required=True)
    p.add_argument('--profiles', nargs='+', choices=FLAGS, default=['smol-base','smol-guided','smol-row','smol-both'])
    args = p.parse_args()
    if args.profiles[0] not in ('baseline','smol-base'):
        p.error('The first profile must be its matched control')
    receipt = json.loads(args.build_info.read_text())
    campaign = transport.Campaign(args.xbox,args.output,receipt,watchdog_seconds=180)
    results=[]
    transport.dump(args.output/'short-protocol.json', {'reply_cap':64,'quality_checks':CHECKS,'profiles':args.profiles,'repetitions':3,'performance_prompt':PERFORMANCE_PROMPT,'watchdog_seconds':180,'model':transport.MODEL,'build':receipt})
    for profile in args.profiles:
        checks=[]
        for label,prompt,expected,cap in CHECKS:
            campaign.reply_token_limit=cap
            r=campaign.run(profile,[{'role':'user','content':prompt}],label)
            if r.get('error'):raise RuntimeError(r['error'])
            ok=transport.structured_checks(r['text'])[0] if expected is None else r['text'].strip()==expected
            checks.append({'label':label,'passed':bool(ok and bounded_record(r,cap)),'text':r['text']})
        campaign.reply_token_limit=64
        runs=[]
        for i in range(3):
            r=campaign.run(profile,[{'role':'user','content':PERFORMANCE_PROMPT}],f'short-{i+1:02}')
            if r.get('error'):raise RuntimeError(r['error'])
            runs.append(r)
        result=evaluate(profile,checks,runs,results[0] if results else None)
        results.append(result);transport.dump(args.output/profile/'short-result.json',result);render(args.output,results)
        print('PROFILE '+profile+': '+json.dumps({k:v for k,v in result.items() if k not in ('checks','runs')}),flush=True)
    if 'smol-base' in args.profiles:
        campaign.reply_token_limit=32
        r=campaign.run('smol-profile',[{'role':'user','content':PERFORMANCE_PROMPT}],'operation-profile')
        if r.get('error'):raise RuntimeError(r['error'])
    print('SHORT CAMPAIGN COMPLETE',flush=True)

if __name__=='__main__':main()
