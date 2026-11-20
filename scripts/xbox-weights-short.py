#!/usr/bin/env python3
# Copyright (c) 2026 xllama contributors
# SPDX-License-Identifier: MIT
"""Serial Ministral weight and paired-matvec trials; all replies <=64 tokens."""
import argparse
import csv
import html
import importlib.util
import json
import statistics
import time
import urllib.request
from pathlib import Path

spec = importlib.util.spec_from_file_location('short', Path(__file__).with_name('xbox-smol-short.py'))
short = importlib.util.module_from_spec(spec)
spec.loader.exec_module(short)
transport = short.transport
# flags, load mode, converted tensor count, exact expected weight-byte reduction.
PROFILES = {
    'smol-base': (0, 0, 0, 0),
    'smol-pair-q4': (8, 0, 0, 0),
    'smol-pair-q6': (16, 0, 0, 0),
    'smol-pair-all': (24, 0, 0, 0),
    'smol-compress-body': (0, 1, 26, 105431040),
    'smol-compress-head': (0, 2, 1, 103809024),
    'smol-compress-all': (0, 3, 27, 209240064),
}
DESCRIPTIONS = {
    'smol-base': ('Control', 'All experimental kernel and weight-load switches off.'),
    'smol-pair-q4': ('Paired Q4 matrix kernel', 'Compute two Q4_K output rows together, sharing Q8 activation loads. Preserve the existing row reduction order and weight layout. The aim is to reduce repeated activation loads, not change matrix arithmetic.'),
    'smol-pair-q6': ('Paired Q6 matrix kernel', 'Apply the same two-row approach only to Q6_K matrices. Odd rows use the stock dot product.'),
    'smol-pair-all': ('Both paired kernels', 'Enable the independently tested Q4_K and Q6_K paired kernels together.'),
    'smol-compress-body': ('Compress transformer weights', 'While loading, requantize the 26 Q6_K transformer matrices to Q4_K. Keep the shared embedding/output matrix at Q6_K. The aim is to reduce resident weight memory and bytes read during multiplication.'),
    'smol-compress-head': ('Compress embedding/output', 'While loading, requantize only the tied token embedding/output matrix from Q6_K to Q4_K. This isolates the memory and multiplication cost of the large output matrix.'),
    'smol-compress-all': ('Compress all Q6 weights', 'Combine the two isolated weight conversions. Existing Q4_K matrices remain unchanged. No vocabulary rows are removed.'),
}

class Campaign(transport.Campaign):
    def activate(self, profile, label):
        request = urllib.request.Request(self.base+'/v1/experiments/reset', data=b'{}', headers={'Content-Type':'application/json'})
        with self.opener.open(request, timeout=30) as response:
            reset = json.load(response)
        if reset.get('reset') is not True:
            raise RuntimeError('The resident model was not reset')
        directory = self.output/profile
        transport.dump(directory/(label+'.preparation.json'), {'reset':reset,'profile':profile,'purpose':'unload, then warm exact profile; GPU selftest is not needed for CPU-only trials'})
        old_guard = self.watchdog_seconds
        self.watchdog_seconds = 600  # Loading/conversion only; reply cap remains eight.
        self.socket_timeout_seconds = 600
        start = time.perf_counter()
        try:
            warm = self.run(profile,[{'role':'user','content':'Reply with exactly: ORCHID'}],label+'.warmup',preparing=True)
        finally:
            self.watchdog_seconds = old_guard
            self.socket_timeout_seconds = 180
        if warm.get('error'):
            raise RuntimeError('Profile load/warmup failed: '+warm['error'])
        transport.dump(directory/'load-warmup.json',{'elapsed_seconds':time.perf_counter()-start,'includes':'model read, any conversion, context creation and one capped warmup','request_cap':warm['request']['max_tokens']})
        self.active_profile = profile


def evaluate(profile, checks, runs, reference=None):
    flags, mode, tensors, saved = PROFILES[profile]
    valid = len(runs)==3 and all(short.bounded_record(r,64) and r['xbox_metrics']['decode_tokens']==64 and r.get('finish_reason')=='length' for r in runs)
    applied = all(r.get('xbox_metrics',{}).get('cpu_experiment_flags')==flags and
                  r['xbox_metrics'].get('weight_compression_mode')==mode and
                  r['xbox_metrics'].get('weight_compressed_tensors')==tensors and
                  r['xbox_metrics'].get('weight_saved_bytes')==saved for r in runs)
    if flags&8:
        applied &= all(r['xbox_metrics'].get('cpu_pair_q4_calls',0)>0 for r in runs)
    if flags&16:
        applied &= all(r['xbox_metrics'].get('cpu_pair_q6_calls',0)>0 for r in runs)
    speed = statistics.median(short.rate(r) for r in runs) if valid else None
    rss = statistics.median(r['xbox_metrics']['sampled_peak_working_set_mib'] for r in runs) if valid else None
    resident = statistics.median(r['xbox_metrics']['resident_model_weight_bytes'] for r in runs) if valid else None
    same = None if reference is None else all(r['text']==reference['runs'][0]['text'] for r in runs)
    repeatable = len({r['text'] for r in runs})==1
    quality = all(c['passed'] for c in checks)
    weights_verified = not mode or bool(reference and resident==reference['resident_model_weight_bytes']-saved)
    speed_win = bool(reference and valid and reference['valid_timing'] and speed>=reference['median_decode_tps']*1.05)
    memory_win = bool(mode and reference and valid and reference['valid_timing'] and rss<=reference['median_rss_mib']*.95 and speed>=reference['median_decode_tps']*.95)
    improved = bool(reference and valid and applied and quality and all(c['passed'] for c in reference['checks']) and repeatable and weights_verified and (mode or same) and (speed_win or memory_win))
    return {'profile':profile,'checks':checks,'runs':runs,'valid_timing':valid,'change_exercised':bool(applied),'resident_weight_reduction_verified':bool(weights_verified),'same_text_as_reference':same,'repeatable_timing_text':repeatable,'median_decode_tps':speed,'median_rss_mib':rss,'resident_model_weight_bytes':resident,'speed_win':speed_win,'memory_win':memory_win,'improved':improved,'scope':'three short timing trials and four content checks; compression can change text; no broad quality or long-response conclusion'}


def table(headers, rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def render(root, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style='<style>body{font:16px system-ui;max-width:1350px;margin:30px auto;padding:20px;line-height:1.5}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #bbb;text-align:left;vertical-align:top}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
    protocol='<p>Same Ministral artifact; CPU only, six workers, context 8192, batch 512, microbatch 128, F16 KV. Three measured 64-token replies per profile, fresh KV and warm weights, temperature zero and seed 42. Content checks and warmups are capped at 8 or 32 tokens. Load/conversion time is separate.</p><p>Improved requires verified execution, three equal-length valid samples, passing content checks, repeatable output, and either at least 5% faster median decode, or (compression only) at least 5% lower observed RSS with speed within 5% of control. Kernel trials additionally require exact control text. Compression trials allow changed text and require the exact expected resident weight reduction. These small checks do not establish broad quality equivalence.</p><p>Compression requantizes existing Q6 weights, so it may add quantization error compared with the original full-precision model. It changes resident RAM weights only; the original GGUF file on disk is untouched. Measured RSS excludes model-loading warmups; it is sampled process working set, not total console RAM.</p>'
    baseline=results[0];summary=[]
    for item in results:
        name=item['profile'];title,change=DESCRIPTIONS[name]
        delta=None if not item['valid_timing'] or not baseline['valid_timing'] else (item['median_decode_tps']/baseline['median_decode_tps']-1)*100
        summary.append([html.escape(title),item['median_decode_tps'],'Unavailable' if delta is None else f'{delta:+.2f}%',item['median_rss_mib'],str(item['improved']).lower(),f'<a href="{name}/report.html">Individual report</a>'])
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        pair=[baseline] if item is baseline else [baseline,item]
        for ax,key,title_plot in [(axes[0],'median_decode_tps','Median engine decode tokens/s'),(axes[1],'median_rss_mib','Median observed RSS MiB')]:
            vals=[x[key] for x in pair]
            if any(v is None for v in vals):ax.text(.5,.5,'Comparable timing unavailable',ha='center',transform=ax.transAxes)
            else:
                bars=ax.bar([x['profile'] for x in pair],vals,color='#5b7891');ax.bar_label(bars,fmt='%.3f',padding=4);ax.set_ylim(0,max(vals+[1])*1.2)
            ax.tick_params(axis='x',labelrotation=15);ax.set_title(title_plot)
        fig.suptitle(title+' | Improved: '+str(item['improved']).lower());fig.savefig(root/name/'comparison.png',dpi=150);fig.savefig(root/name/'comparison.svg');plt.close(fig)
        details={k:v for k,v in item.items() if k not in ('checks','runs')}
        samples=[]
        for i,r in enumerate(item['runs'],1):
            m=r['xbox_metrics'];samples.append([i,f'{short.rate(r):.6f}',m['decode_tokens'],r['finish_reason'],m['sampled_peak_working_set_mib'],m['resident_model_weight_bytes'],m['weight_compressed_tensors'],m['weight_saved_bytes'],m['cpu_pair_q4_calls'],m['cpu_pair_q6_calls']])
        checks=table(['Check','Content passed','Strict format passed','Full answer'],[[html.escape(c['label']),str(c['passed']).lower(),str(c.get('format_passed','Not applicable')).lower(),'<pre>'+html.escape(c['text'])+'</pre>'] for c in item['checks']])
        outputs=''.join('<h3>Timing response '+str(i)+'</h3><pre>'+html.escape(r['text'])+'</pre>' for i,r in enumerate(item['runs'],1))
        load=json.loads((root/name/'load-warmup.json').read_text())
        doc='<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title>'+style+'<h1>'+html.escape(title)+'</h1><p>'+html.escape(change)+'</p>'+protocol+'<img src="comparison.png">'+table(['Metric','Outcome'],[[html.escape(k),html.escape(str(v))] for k,v in details.items()])+ '<h2>Each repetition</h2>'+table(['Run','Decode tok/s','Tokens','Finish','RSS MiB','Resident weight bytes','Converted tensors','Weight bytes saved','Q4 pair calls','Q6 pair calls'],samples)+'<h2>Content checks</h2>'+checks+'<h2>Model load and warmup</h2><pre>'+html.escape(json.dumps(load,indent=2))+'</pre>'+outputs+'<p><a href="short-result.json">Structured result</a> · <a href="../report.html">All experiments</a></p>'
        (root/name/'report.html').write_text(doc)
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    measured=[x for x in results if x['median_decode_tps'] is not None]
    if measured:
        bars=ax.bar(range(len(measured)),[x['median_decode_tps'] for x in measured],color=['#24864f' if x['improved'] else '#5b7891' for x in measured])
        for i,x in enumerate(measured):ax.scatter([i-.1,i,i+.1],[short.rate(r) for r in x['runs']],color='#183347',s=20)
        ax.bar_label(bars,fmt='%.3f',padding=-28,color='white');ax.set_xticks(range(len(measured)),[x['profile'] for x in measured],rotation=20,ha='right');ax.set_ylim(0,max(short.rate(r) for x in measured for r in x['runs'])*1.2)
    ax.set(ylabel='Engine decode tokens / second',title='Ministral | weight compression and paired matrix kernels');fig.savefig(root/'comparison.png',dpi=150);fig.savefig(root/'comparison.svg');plt.close(fig)
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Ministral weight and matrix experiments</title>'+style+'<h1>Ministral weight and matrix experiments</h1>'+protocol+table(['Test','Median tok/s','Vs control','RSS MiB','Improved','Evidence'],summary)+'<img src="comparison.png"><p><a href="results.json">Structured results</a> · <a href="protocol.json">Protocol and console receipt</a></p>')
    transport.dump(root/'results.json',results)
    columns=['profile','median_decode_tps','median_rss_mib','resident_model_weight_bytes','change_exercised','same_text_as_reference','repeatable_timing_text','speed_win','memory_win','improved']
    with (root/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows({k:item[k] for k in columns} for item in results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xbox',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--build-info',required=True,type=Path)
    args=p.parse_args();receipt=json.loads(args.build_info.read_text())
    campaign=Campaign(args.xbox,args.output,receipt,watchdog_seconds=180)
    with campaign.opener.open(campaign.base+'/v1/experiments/info',timeout=15) as response:info=json.load(response)
    if info.get('weight_experiment_protocol')!=1 or info.get('version')!=receipt['required_console_version'] or info.get('app_memory_limit_bytes',0)<5*1024**3:
        raise RuntimeError('Version/protocol/Game memory preflight failed: '+str(info))
    transport.dump(args.output/'protocol.json',{'profiles':PROFILES,'console':info,'checks':short.CHECKS,'performance_prompt':short.PERFORMANCE_PROMPT,'max_reply_tokens':64,'repetitions':3,'improvement_rule':'5% faster median decode OR compression 5% lower RSS and speed within5%; passing content checks; proven path; exact kernel output; deterministic compression output. No general quality equivalence.'})
    results=[]
    for profile in PROFILES:
        checks=[]
        for label,prompt,expected,cap in short.CHECKS:
            campaign.reply_token_limit=cap
            r=campaign.run(profile,[{'role':'user','content':prompt}],label)
            if r.get('error'):raise RuntimeError(r['error'])
            passed,strict=transport.structured_checks(r['text']) if expected is None else (r['text'].strip()==expected,None)
            checks.append({'label':label,'passed':bool(passed and short.bounded_record(r,cap)),'format_passed':strict,'text':r['text']})
        campaign.reply_token_limit=64
        runs=[]
        for i in range(3):
            r=campaign.run(profile,[{'role':'user','content':short.PERFORMANCE_PROMPT}],f'short-{i+1:02}')
            if r.get('error'):raise RuntimeError(r['error'])
            runs.append(r)
        item=evaluate(profile,checks,runs,results[0] if results else None)
        results.append(item);transport.dump(args.output/profile/'short-result.json',item);render(args.output,results)
        print('PROFILE '+profile+': '+json.dumps({k:v for k,v in item.items() if k not in ('checks','runs')}),flush=True)
    print('WEIGHT AND MATRIX SHORT CAMPAIGN COMPLETE',flush=True)

if __name__=='__main__':main()
