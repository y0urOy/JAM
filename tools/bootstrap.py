#!/usr/bin/env python3
"""Install an isolated backend environment; never reuse an existing research env."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('profile',choices=['wan','ovis','flux','qwen'])
    ap.add_argument('--venv',required=True)
    ap.add_argument('--python',default=sys.executable)
    ap.add_argument('--torch-index',default='https://download.pytorch.org/whl/cu128')
    ap.add_argument('--libero',action='store_true')
    ap.add_argument('--fast-kernels',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args();venv=Path(a.venv).expanduser().resolve();py=str(venv/'bin/python')
    backend='wan' if a.profile=='wan' else 'multimodal'
    ds='0.19.2' if a.profile=='ovis' else '0.18.5'
    commands=[
        [a.python,'-m','venv',str(venv)],
        [py,'-m','pip','install','--upgrade','pip','setuptools','wheel'],
        [py,'-m','pip','install','torch==2.7.1','torchvision==0.22.1','--index-url',a.torch_index],
        [py,'-m','pip','install','-r',str(ROOT/f'environments/{a.profile}.txt'),
         '-c',str(ROOT/f'environments/constraints_{a.profile}.txt')],
        [py,'-m','pip','install','--no-build-isolation',f'deepspeed=={ds}',
         '-c',str(ROOT/f'environments/constraints_{a.profile}.txt')],
        [py,'-m','pip','install','--no-deps','-e',str(ROOT/f'backends/{backend}')],
    ]
    if a.libero:commands += [[py,'-m','pip','install','-r',str(ROOT/'environments/libero.txt'),
                               '-c',str(ROOT/f'environments/constraints_{a.profile}.txt')]]
    if a.fast_kernels:
        if a.profile!='qwen':raise SystemExit('--fast-kernels is a Qwen option.')
        commands += [[py,'-m','pip','install','--no-build-isolation','-r',str(ROOT/'environments/qwen-fast-kernels.txt')]]
    commands += [[py,'-m','pip','check']]
    import shlex
    if a.dry_run:
        print('\n'.join(shlex.join(c) for c in commands));return
    if venv.exists():raise SystemExit('Choose a new venv path; existing environments are not overwritten.')
    version=subprocess.check_output([a.python,'-c','import sys; print(sys.version_info[0],sys.version_info[1])'],text=True).strip()
    if tuple(map(int,version.split()))!=(3,10):raise SystemExit('Use Python 3.10 to match the recorded runtime.')
    env=os.environ.copy();env.setdefault('DS_BUILD_OPS','0')
    for c in commands:
        print(shlex.join(c),flush=True);subprocess.run(c,check=True,env=env)

if __name__=='__main__':main()
