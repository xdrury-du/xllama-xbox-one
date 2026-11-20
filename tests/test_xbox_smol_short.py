# Copyright (c) 2026 xllama contributors
# SPDX-License-Identifier: MIT
import importlib.util
import unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('smol_short',Path(__file__).resolve().parents[1]/'scripts/xbox-smol-short.py')
short=importlib.util.module_from_spec(spec);spec.loader.exec_module(short)

def run(rate=3,flags=0,text='same'):
 return {'request':{'max_tokens':64},'done':True,'error':None,'finish_reason':'length','text':text,
         'xbox_metrics':{'decode_tokens':64,'decode_ms':64000/rate,'reply_token_budget':64,
         'auto_reply_budget':False,'cpu_experiment_flags':flags,'cpu_guided_calls':10 if flags&2 else 0,
         'cpu_row_calls':10 if flags&4 else 0}}

class ShortVerdictTests(unittest.TestCase):
 def test_explicit_cap_and_actual_budget_are_required(self):
  r=run();self.assertTrue(short.bounded_record(r,64))
  r['request'].pop('max_tokens');self.assertFalse(short.bounded_record(r,64))
  r=run();r['xbox_metrics']['decode_tokens']=65;self.assertFalse(short.bounded_record(r,64))
 def test_early_stop_cannot_count_as_equal_length_speedup(self):
  r=run();r['finish_reason']='stop';r['xbox_metrics']['decode_tokens']=12
  self.assertFalse(short.evaluate('smol-base',[{'passed':True}],[r]*3)['valid_timing'])
 def test_gain_requires_actual_execution_and_matching_text(self):
  base=short.evaluate('smol-base',[{'passed':True}],[run()]*3)
  self.assertTrue(short.evaluate('smol-guided',[{'passed':True}],[run(3.3,2)]*3,base)['improved'])
  r=run(3.3,2);r['xbox_metrics']['cpu_guided_calls']=0
  self.assertFalse(short.evaluate('smol-guided',[{'passed':True}],[r]*3,base)['improved'])
  self.assertFalse(short.evaluate('smol-guided',[{'passed':True}],[run(3.3,2,'different')]*3,base)['improved'])
 def test_failure_and_subthreshold_gain_stay_false(self):
  base=short.evaluate('smol-base',[{'passed':True}],[run()]*3)
  self.assertFalse(short.evaluate('smol-row',[{'passed':False}],[run(3.3,4)]*3,base)['improved'])
  self.assertFalse(short.evaluate('smol-row',[{'passed':True}],[run(3.1,4)]*3,base)['improved'])
