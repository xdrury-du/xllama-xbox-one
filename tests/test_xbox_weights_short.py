# Copyright (c) 2026 xllama contributors
# SPDX-License-Identifier: MIT
import copy
import importlib.util
import unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('weights_short',Path(__file__).resolve().parents[1]/'scripts/xbox-weights-short.py')
weights=importlib.util.module_from_spec(spec);spec.loader.exec_module(weights)

def run(profile='smol-base',rate=3,rss=3000,text='same'):
 flags,mode,tensors,saved=weights.PROFILES[profile]
 return {'request':{'max_tokens':64},'done':True,'error':None,'finish_reason':'length','text':text,
 'xbox_metrics':{'decode_tokens':64,'decode_ms':64000/rate,'reply_token_budget':64,'auto_reply_budget':False,
 'cpu_experiment_flags':flags,'weight_compression_mode':mode,'weight_compressed_tensors':tensors,'weight_saved_bytes':saved,
 'cpu_pair_q4_calls':10 if flags&8 else 0,'cpu_pair_q6_calls':10 if flags&16 else 0,'sampled_peak_working_set_mib':rss,
 'resident_model_weight_bytes':2138615808-saved}}

class WeightVerdictTests(unittest.TestCase):
 def setUp(self):
  self.checks=[{'passed':True}]
  self.base=weights.evaluate('smol-base',self.checks,[run()]*3)
 def test_kernel_win_requires_counter_and_identical_text(self):
  self.assertTrue(weights.evaluate('smol-pair-q4',self.checks,[run('smol-pair-q4',3.3)]*3,self.base)['improved'])
  r=run('smol-pair-q4',3.3);r['xbox_metrics']['cpu_pair_q4_calls']=0
  self.assertFalse(weights.evaluate('smol-pair-q4',self.checks,[r]*3,self.base)['improved'])
  self.assertFalse(weights.evaluate('smol-pair-q4',self.checks,[run('smol-pair-q4',3.3,text='different')]*3,self.base)['improved'])
 def test_compression_can_change_text_but_must_prove_exact_resident_reduction(self):
  r=run('smol-compress-all',3,2800,'changed')
  self.assertTrue(weights.evaluate('smol-compress-all',self.checks,[r]*3,self.base)['improved'])
  r['xbox_metrics']['resident_model_weight_bytes']+=64
  self.assertFalse(weights.evaluate('smol-compress-all',self.checks,[r]*3,self.base)['improved'])
 def test_quality_failure_slowdown_or_wrong_conversion_cannot_win(self):
  r=run('smol-compress-all',3,2800)
  self.assertFalse(weights.evaluate('smol-compress-all',[{'passed':False}],[r]*3,self.base)['improved'])
  self.assertFalse(weights.evaluate('smol-compress-all',self.checks,[run('smol-compress-all',2.8,2800)]*3,self.base)['improved'])
  r['xbox_metrics']['weight_compressed_tensors']=26
  self.assertFalse(weights.evaluate('smol-compress-all',self.checks,[r]*3,self.base)['improved'])
 def test_early_eos_or_inconsistent_output_prevents_win(self):
  r=run('smol-compress-all',3.3,2800);r['xbox_metrics']['decode_tokens']=60;r['finish_reason']='stop'
  self.assertFalse(weights.evaluate('smol-compress-all',self.checks,[r]*3,self.base)['improved'])
  samples=[run('smol-compress-all',3.3,2800) for _ in range(3)];samples[1]['text']='changed'
  self.assertFalse(weights.evaluate('smol-compress-all',self.checks,samples,self.base)['improved'])
 def test_memory_improvement_threshold_and_all_three_repetitions_are_required(self):
  self.assertFalse(weights.evaluate('smol-compress-body',self.checks,[run('smol-compress-body',3,2900)]*3,self.base)['improved'])
  self.assertFalse(weights.evaluate('smol-pair-all',self.checks,[run('smol-pair-all',3.3)]*2,self.base)['improved'])
 def test_replies_are_explicitly_capped_in_the_measured_protocol(self):
  r=run('smol-pair-q6',3.3);r['request']['max_tokens']=65
  self.assertFalse(weights.evaluate('smol-pair-q6',self.checks,[r]*3,self.base)['improved'])
