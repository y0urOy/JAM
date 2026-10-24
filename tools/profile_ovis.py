#!/usr/bin/env python3
"""Plan Ovis measured GPU-time variants using the preserved trainer instrumentation."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from jam import parser, command_plan

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('variant',choices=['generation','online','cached-emulated'])
    ap.add_argument('--data-root');ap.add_argument('--model-root');ap.add_argument('--cache-root')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    name='ovis-generation' if a.variant=='generation' else 'ovis-jam'
    args=parser().parse_args(['train',name,'--micro-batch','16','--grad-accum','8'])
    args.data_root=a.data_root;args.model_root=a.model_root;args.cache_root=a.cache_root
    # Instrument one GPU at microbatch 16. This is a 100-update timing probe,
    # not a reproduction training run or an evaluation checkpoint.
    args.override=['++max_steps=100','++save_every=0','++save_final_checkpoint=false']
    plan,env=command_plan(args)
    # one optimizer update per micro-batch; the launcher manages accumulation, so set it after planning
    plan['commands']=[[('++gradient_accumulation_steps=1' if str(x).startswith('++gradient_accumulation_steps=') else x) for x in c]
                      for c in plan['commands']]
    env['JAM_PROFILE_GPU_STEPS']='20:80'
    env['JAM_FORESIGHT_CACHE_EMULATE']='1' if a.variant=='cached-emulated' else '0'
    plan['timing_micro_batch']=16
    plan['timing_global_batch']=16
    if a.dry_run:
        import json
        plan['timing_env']={k:env[k] for k in ('JAM_PROFILE_GPU_STEPS','JAM_FORESIGHT_CACHE_EMULATE')}
        plan['note']='Cached-emulated reuses first-batch targets for timing only. Never use this mode to train/report a policy.'
        print(json.dumps(plan,indent=2));return
    for c in plan['commands']:subprocess.run(c,cwd=plan['cwd'],env=env,check=True)

if __name__=='__main__':main()
