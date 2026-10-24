#!/usr/bin/env python3
"""Verify every released file and internal symlink without loading model assets."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'FILE_MANIFEST.json').read_text())
    errors=[]
    for item in manifest['files']:
        p=root/item['path']
        if 'symlink' in item:
            import os
            if not p.is_symlink() or os.readlink(p)!=item['symlink']:errors.append(item['path'])
            try:p.resolve().relative_to(root)
            except ValueError:errors.append(item['path']+' (external link)')
        elif not p.is_file() or p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            errors.append(item['path'])
    if errors:raise SystemExit('Manifest mismatch: '+', '.join(errors))
    print(f'Verified {len(manifest["files"])} released entries.')

if __name__=='__main__':main()
