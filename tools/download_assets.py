#!/usr/bin/env python3
"""Download one explicitly selected public asset from a pinned official source."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('asset');p.add_argument('--model-root');p.add_argument('--data-root')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();assets=json.loads((ROOT/'reproduction/assets.json').read_text())['assets']
    if a.asset not in assets:raise SystemExit('Choose an asset: '+', '.join(assets))
    item=assets[a.asset]
    root=Path(a.data_root or os.environ.get('JAM_DATA_ROOT',ROOT/'artifacts/data')) if item.get('repo_type')=='dataset' else Path(a.model_root or os.environ.get('JAM_MODEL_ROOT',ROOT/'artifacts/models'))
    if not item.get('revision'):raise SystemExit('No verified upstream revision is recorded for this asset; see docs/DATA_AND_MODELS.md.')
    kwargs=dict(repo_id=item['repo_id'],revision=item['revision'],repo_type=item.get('repo_type','model'),
                local_dir=str(root/item['directory']))
    if item.get('allow_patterns'):kwargs['allow_patterns']=item['allow_patterns']
    if a.dry_run:print(json.dumps(kwargs,indent=2));return
    from huggingface_hub import snapshot_download
    snapshot_download(**kwargs)
    print('Downloaded',item['repo_id'],'at',item['revision'],'to',kwargs['local_dir'])
    print('Dataset archives require extraction; see docs/DATA_AND_MODELS.md.')
if __name__=='__main__':main()
