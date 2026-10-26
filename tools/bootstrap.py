#!/usr/bin/env python3
"""Install the pinned JAM environment in a new venv; never reuse an existing research env."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/'environments'
COMPILED='# Compiled against the installed PyTorch'

def requirements():
    """Split requirements.txt into wheels and the packages built against the installed PyTorch."""
    lines=[line.strip() for line in (ENV/'requirements.txt').read_text().splitlines()]
    split=lines.index(COMPILED)
    keep=lambda block:[line for line in block if line and not line.startswith('#')]
    return keep(lines[:split]),keep(lines[split+1:])

def build_environment():
    # DeepSpeed compiles its ops just in time. On aarch64, compile flash-attn and causal-conv1d
    # from source, as in the recorded environment, instead of looking for a wheel on GitHub.
    env={'DS_BUILD_OPS':'0'}
    if platform.machine() in ('aarch64','arm64'):
        env.update(FLASH_ATTENTION_FORCE_BUILD='TRUE',CAUSAL_CONV1D_FORCE_BUILD='TRUE')
    return env

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--venv',required=True)
    ap.add_argument('--python',default=sys.executable)
    ap.add_argument('--torch-index',default='https://download.pytorch.org/whl/cu128')
    ap.add_argument('--libero',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args();venv=Path(a.venv).expanduser().resolve();py=str(venv/'bin/python')
    wheels,compiled=requirements();constraints=['-c',str(ENV/'constraints.txt')]
    commands=[
        [a.python,'-m','venv',str(venv)],
        [py,'-m','pip','install','--upgrade','pip','setuptools','wheel',*constraints],
        [py,'-m','pip','install','torch==2.7.1','torchvision==0.22.1','--index-url',a.torch_index],
        [py,'-m','pip','install',*wheels,*constraints],
        [py,'-m','pip','install','--no-deps','-e',str(ROOT/'backends/wan'),'-e',str(ROOT/'backends/multimodal')],
    ]
    if a.libero:commands += [[py,'-m','pip','install','-r',str(ENV/'libero.txt'),*constraints]]
    commands += [[py,'-m','pip','install','--no-build-isolation',*compiled,*constraints],
                 [py,'-m','pip','check']]
    build=build_environment()
    import shlex
    print('# build environment: '+' '.join(f'{k}={v}' for k,v in build.items()),flush=True)
    if a.dry_run:
        print('\n'.join(shlex.join(c) for c in commands));return
    if venv.exists():raise SystemExit('Choose a new venv path; existing environments are not overwritten.')
    version=subprocess.check_output([a.python,'-c','import sys; print(sys.version_info[0],sys.version_info[1])'],text=True).strip()
    if tuple(map(int,version.split()))!=(3,10):raise SystemExit('Use Python 3.10 to match the recorded runtime.')
    env=os.environ.copy()
    for key,value in build.items():env.setdefault(key,value)
    for c in commands:
        print(shlex.join(c),flush=True);subprocess.run(c,check=True,env=env)

if __name__=='__main__':main()
