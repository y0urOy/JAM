#!/usr/bin/env python3
"""Generate action initializations from the corresponding public pretrained backbone."""
from __future__ import annotations
import argparse,os,subprocess,sys,json
from pathlib import Path
from jam import ROOT,environment,paths

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('family',choices=['wan','ovis','flux']);p.add_argument('--action-dim',type=int,choices=[7,14],default=7)
    p.add_argument('--model-root');p.add_argument('--data-root');p.add_argument('--cache-root')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();ps=paths(a);models=ps['models'];backend='wan' if a.family=='wan' else 'multimodal'
    repo=ROOT/'backends'/backend
    if a.family=='wan':
        dest=models/'init/wan_action.pt'
        cmd=[sys.executable,'scripts/preprocess_action_dit_backbone.py','--model-config','configs/model/jam_wan.yaml','--output',str(dest),'--device','cpu']
    elif a.family=='ovis':
        dest=models/f'init/ovis_action_{a.action_dim}d.pt'
        cmd=[sys.executable,'scripts/ovis_u1/preprocess_action_dit_ovis_u1.py','--model-config','configs/model/jam_ovis_u1.yaml',
             '--ovis-u1-model-path',str(models/'Ovis-U1-3B'),'--output',str(dest),'--action-dim',str(a.action_dim),'--device','cpu']
    else:
        dest=models/f'init/flux_action_{a.action_dim}d.pt'
        cmd=[sys.executable,'scripts/flux2/preprocess_action_dit_flux2.py','--model-config','configs/model/jam_flux2_klein_4b_base.yaml',
             '--flux2-src-path',str(repo/'third_party/flux2'),'--flux2-model-path',str(models/'flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors'),
             '--output',str(dest),'--action-dim',str(a.action_dim),'--device','cpu']
    if a.dry_run:print(json.dumps({'cwd':str(repo),'command':cmd,'output':str(dest)},indent=2));return
    if dest.exists():raise SystemExit('Initialization already exists; verify it before using it.')
    dest.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(cmd,cwd=repo,env=environment(backend,'libero',ps),check=True)
if __name__=='__main__':main()
