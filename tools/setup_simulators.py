#!/usr/bin/env python3
"""Fetch a pinned LIBERO checkout and install its Python package, without running a simulator."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'libero':('https://github.com/Lifelong-Robot-Learning/LIBERO.git','8f1084e3132a39270c3a13ebe37270a43ece2a01','LIBERO'),
 'plus':('https://github.com/sylvestf/LIBERO-plus.git','4976dc30028e805ff8094b55501d532c48fec182','LIBERO-plus'),
}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('benchmark',choices=list(SOURCES))
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--skip-install',action='store_true')
    a=ap.parse_args();url,rev,name=SOURCES[a.benchmark];dest=ROOT/'third_party'/name
    commands=[['git','clone','--no-checkout',url,str(dest)],['git','-C',str(dest),'checkout','--detach',rev]]
    patch=ROOT/'patches'/f'{name}.patch'
    if patch.is_file():commands += [['git','-C',str(dest),'apply',str(patch)]]
    if not a.skip_install:commands += [[sys.executable,'-m','pip','install','--no-deps','-e',str(dest)]]
    import shlex
    if a.dry_run:print('\n'.join(shlex.join(c) for c in commands));return
    if dest.exists():raise SystemExit('Checkout exists; inspect it rather than overwriting it.')
    dest.parent.mkdir(parents=True,exist_ok=True)
    for c in commands:subprocess.run(c,check=True)

if __name__=='__main__':main()
