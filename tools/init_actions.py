#!/usr/bin/env python3
"""Generate action initializations from the corresponding public pretrained backbone."""
from __future__ import annotations
import argparse,os,subprocess,sys,json
from pathlib import Path
from jam import ROOT,environment,paths

def require_wan_initialization_assets(models):
    backbone=models/'Wan-AI/Wan2.2-TI2V-5B'
    tokenizer=models/'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'
    required=[backbone/f'diffusion_pytorch_model-{i:05d}-of-00003.safetensors' for i in range(1,4)]
    required += [backbone/'Wan2.2_VAE.pth',backbone/'models_t5_umt5-xxl-enc-bf16.pth']
    required += [tokenizer/name for name in ('special_tokens_map.json','spiece.model','tokenizer.json','tokenizer_config.json')]
    missing=[str(path) for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise SystemExit('Missing Wan initialization assets:\n  '+'\n  '.join(missing)
                         +'\nDownload wan and wan-tokenizer with tools/download_assets.py into the same --model-root.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('family',choices=['wan','ovis','flux']);p.add_argument('--action-dim',type=int,choices=[7,14],default=7)
    p.add_argument('--model-root');p.add_argument('--data-root');p.add_argument('--cache-root')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();ps=paths(a);models=ps['models'];backend='wan' if a.family=='wan' else 'multimodal'
    repo=ROOT/'backends'/backend
    if a.family=='wan':
        dest=models/'init/wan_action.pt'
        # BF16 on CUDA, as the original initialization was made
        cmd=[sys.executable,'scripts/preprocess_action_dit_backbone.py','--model-config','configs/model/jam_wan.yaml','--output',str(dest),'--device','cuda','--dtype','bfloat16']
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
    if a.family=='wan':require_wan_initialization_assets(models)
    dest.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(cmd,cwd=repo,env=environment(backend,'libero',ps),check=True)
if __name__=='__main__':main()
