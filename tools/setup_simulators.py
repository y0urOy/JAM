#!/usr/bin/env python3
"""Prepare a simulator without running it: the pinned LIBERO or LIBERO-Plus checkout (clone, patch, assets,
configuration), or the installed RoboTwin 2.0 simulator (assets, source edits, embodiment paths)."""
from __future__ import annotations
import argparse
from importlib import metadata, util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from download_assets import sha256, verify_files
from jam import ROOT, libero_config_dir
SOURCES={
 'libero':('https://github.com/Lifelong-Robot-Learning/LIBERO.git','8f1084e3132a39270c3a13ebe37270a43ece2a01','LIBERO'),
 'plus':('https://github.com/sylvestf/LIBERO-plus.git','4976dc30028e805ff8094b55501d532c48fec182','LIBERO-plus'),
}
# Archives fetched by tools/download_assets.py into the data root.
LIBERO_PLUS_ASSETS='libero-plus-assets/assets.zip'
ROBOTWIN_ASSETS=[f'robotwin-assets/{name}.zip' for name in ('background_texture','embodiments','objects')]
# The SAPIEN and MPlib source edits that RoboTwin's script/_install.sh describes (old -> new), as applied in the
# paper's evaluation environment.
ROBOTWIN_EDITS={
 ('sapien','wrapper/urdf_loader.py'):[
  ('with open(urdf_file, "r") as f:','with open(urdf_file, "r", encoding="utf-8") as f:'),
  ('urdf_file[:-4] + "srdf"','urdf_file[:-4] + ".srdf"'),
  ('with open(srdf_file, "r") as f:','with open(srdf_file, "r", encoding="utf-8") as f:')],
 ('mplib','planner.py'):[
  ('if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:',
   'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:')],
}

def data_root(a):
    return Path(a.data_root or os.environ.get('JAM_DATA_ROOT',ROOT/'artifacts/data')).expanduser().resolve()

def run(command,dry_run,cwd=None):
    print(('cd '+shlex.quote(str(cwd))+' && ' if cwd else '')+shlex.join(command))
    if not dry_run:subprocess.run(command,check=True,cwd=cwd)

def extract_assets(archive,dest,dry_run,component=None):
    """Publish a complete extracted directory; failed extraction leaves no partial destination."""
    print(f'extract {archive} -> {dest}')
    if dry_run:return
    if dest.exists():raise ValueError(f'{dest} already exists; move the incomplete asset directory aside before extracting again.')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.jam-assets-',dir=dest.parent) as temporary:
        staged=Path(temporary)/'contents';staged.mkdir();count=0
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():continue
                parts=info.filename.split('/')
                if component is None:
                    # LIBERO-Plus nests assets/ under its original download path.
                    if 'assets' not in parts:continue
                    rel=parts[parts.index('assets')+1:]
                else:
                    if parts[0]!=component:continue
                    rel=parts[1:]
                if not rel or any(x in ('','..','.') for x in rel):
                    raise ValueError(f'Invalid asset archive path: {info.filename}')
                target=staged.joinpath(*rel);target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as source,target.open('wb') as out:shutil.copyfileobj(source,out)
                count+=1
        if not count:raise ValueError(f'No {component or "assets"} files found in {archive}.')
        staged.rename(dest)

def asset_metadata(key):
    return json.loads((ROOT/'reproduction/assets.json').read_text())['assets'][key]

def verify_extracted_assets(key,directory,component=None):
    """Check the simulator files whose versions affect the recorded evaluation."""
    for name,expected in asset_metadata(key).get('extracted_sha256',{}).items():
        if component and not name.startswith(component+'/'):continue
        path=directory/name
        if not path.is_file() or sha256(path)!=expected:
            raise ValueError(f'Missing or incompatible simulator asset: {path}; '
                             'move the incomplete asset directory aside, download the pinned assets and run setup again.')

def write_libero_config(benchmark,dest,dry_run=False):
    root=dest/'libero'/'libero'
    paths={'benchmark_root':root,'bddl_files':root/'bddl_files','init_states':root/'init_files',
           'datasets':dest/'libero'/'datasets','assets':root/'assets'}
    config=libero_config_dir(benchmark)/'config.yaml'
    print(f'write {config}')
    if dry_run:return
    config.parent.mkdir(parents=True,exist_ok=True);paths['datasets'].mkdir(parents=True,exist_ok=True)
    config.write_text(''.join(f'{k}: {json.dumps(str(v))}\n' for k,v in paths.items()))

def prepare_libero(a):
    url,rev,name=SOURCES[a.benchmark];dest=ROOT/'third_party'/name;patch=ROOT/'patches'/f'{name}.patch'
    fresh=not dest.exists()
    if fresh:
        if a.dry_run:
            run(['git','clone','--no-checkout',url,str(dest)],True)
            run(['git','-C',str(dest),'checkout','--detach',rev],True)
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.jam-clone-',dir=dest.parent) as temporary:
                checkout=Path(temporary)/'checkout'
                run(['git','clone','--no-checkout',url,str(checkout)],False)
                run(['git','-C',str(checkout),'checkout','--detach',rev],False)
                checkout.rename(dest)
    else:
        head=subprocess.run(['git','-C',str(dest),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
        if head!=rev:raise SystemExit(f'{dest} is at {head or "an unknown revision"}, not {rev}; move it aside and run this again.')
    if not fresh and subprocess.run(['git','-C',str(dest),'apply','-R','--check',str(patch)],capture_output=True).returncode==0:
        print(f'{patch.name} already applied')
    else:
        run(['git','-C',str(dest),'apply',str(patch)],a.dry_run)
    assets=dest/'libero'/'libero'/'assets'
    if a.benchmark=='plus' and not assets.is_dir():
        archive=data_root(a)/LIBERO_PLUS_ASSETS
        if not archive.is_file():raise SystemExit('Download the LIBERO-Plus assets first: python tools/download_assets.py libero-plus-assets')
        if not a.dry_run:verify_files(archive.parent,asset_metadata('libero-plus-assets')['files'])
        extract_assets(archive,assets,a.dry_run)
    if a.benchmark=='plus' and (not a.dry_run or assets.exists()):
        verify_extracted_assets('libero-plus-assets',assets)
    write_libero_config(a.benchmark,dest,a.dry_run)

def require_paper_sapien():
    try:
        version=metadata.version('sapien')
    except metadata.PackageNotFoundError:
        raise SystemExit('Install the RoboTwin simulator packages first; see docs/INSTALL.md.') from None
    if version!='3.0.0b1':
        if os.environ.get('JAM_ALLOW_SAPIEN_VERSION')!=version:
            raise SystemExit(f'RoboTwin 2.0 results need SAPIEN 3.0.0b1, found {version}; see docs/INSTALL.md.')
        print(f'SAPIEN {version}: success rates are not comparable with the paper (JAM_ALLOW_SAPIEN_VERSION).')

def prepare_robotwin(a):
    require_paper_sapien()
    robotwin=ROOT/'third_party'/'RoboTwin';assets=robotwin/'assets'
    for archive_name in ROBOTWIN_ASSETS:
        component=Path(archive_name).stem
        if not (assets/component).is_dir():
            archive=data_root(a)/archive_name
            if not archive.is_file():
                raise SystemExit('Download the RoboTwin assets first: python tools/download_assets.py robotwin-assets')
            if not a.dry_run:
                verify_files(archive.parent,{archive.name:asset_metadata('robotwin-assets')['files'][archive.name]})
            extract_assets(archive,assets/component,a.dry_run,component=component)
        if not a.dry_run or (assets/component).exists():
            verify_extracted_assets('robotwin-assets',assets,component)
    for (package,name),edits in ROBOTWIN_EDITS.items():
        spec=util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit(f'{package} is not installed; see docs/INSTALL.md.')
        path=Path(spec.submodule_search_locations[0])/name;original=path.read_text(encoding='utf-8');text=original
        for old,new in edits:
            if old in text:text=text.replace(old,new)
            elif new not in text:raise SystemExit(f'Cannot find the RoboTwin edit point in {path}: {old}')
        if text==original:
            print(f'already applied {path}');continue
        print(f'{"would patch" if a.dry_run else "patched"} {path}')
        if not a.dry_run:
            try:
                path.write_text(text,encoding='utf-8')
            except OSError as error:
                raise SystemExit(f'Cannot write {path} ({error}); run this with write access to that environment.') from None
    run([sys.executable,'script/update_embodiment_config_path.py'],a.dry_run,cwd=robotwin)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('benchmark',choices=list(SOURCES)+['robotwin'])
    ap.add_argument('--data-root',help='Where tools/download_assets.py put the simulator assets.')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    try:
        prepare_robotwin(a) if a.benchmark=='robotwin' else prepare_libero(a)
    except (OSError,ValueError,zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(f'{shlex.join(error.cmd)} failed with exit code {error.returncode}.') from None

if __name__=='__main__':main()
