"""Archive integrity and interrupted simulator preparation, without loading a simulator."""
from pathlib import Path
import argparse
import hashlib
import subprocess
import sys
import zipfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import download_assets
import setup_simulators
import init_actions


def write_archive(path,members):
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w') as archive:
        for name,data in members.items():archive.writestr(name,data)
    return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def test_archive_checksum_detects_same_size_corruption(tmp_path):
    path=tmp_path/'assets.zip'
    checks=write_archive(path,{'assets/test.xml':b'<asset/>'})
    download_assets.verify_files(tmp_path,{path.name:checks})
    data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match='checksum mismatch'):
        download_assets.verify_files(tmp_path,{path.name:checks})


@pytest.mark.parametrize('component,prefix',[(None,'old/download/assets/'),('objects','objects/')])
def test_extracted_assets_have_portable_paths(tmp_path,component,prefix):
    archive=tmp_path/'input.zip';target=tmp_path/'installed'
    write_archive(archive,{prefix+'scene/file.xml':b'<scene/>'})
    setup_simulators.extract_assets(archive,target,False,component=component)
    assert (target/'scene/file.xml').read_bytes()==b'<scene/>'
    assert sorted(p.name for p in tmp_path.iterdir())==['input.zip','installed']


def test_failed_extraction_cleans_partial_files_and_can_retry(tmp_path,monkeypatch):
    archive=tmp_path/'input.zip';target=tmp_path/'installed'
    write_archive(archive,{'assets/first.xml':b'first','assets/second.xml':b'second'})
    original=zipfile.ZipFile.open
    def fail_second(self,name,*args,**kwargs):
        if getattr(name,'filename',name).endswith('second.xml'):raise OSError('read interrupted')
        return original(self,name,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipFile,'open',fail_second)
        with pytest.raises(OSError,match='read interrupted'):
            setup_simulators.extract_assets(archive,target,False)
    assert not target.exists()
    assert not list(tmp_path.glob('.jam-assets-*'))
    setup_simulators.extract_assets(archive,target,False)
    assert (target/'first.xml').read_bytes()==b'first'
    assert (target/'second.xml').read_bytes()==b'second'


def test_invalid_archive_path_leaves_no_published_or_escaped_files(tmp_path):
    archive=tmp_path/'input.zip';target=tmp_path/'installed'
    write_archive(archive,{'assets/first.xml':b'first','assets/../escape.txt':b'bad'})
    with pytest.raises(ValueError,match='Invalid asset archive path'):
        setup_simulators.extract_assets(archive,target,False)
    assert sorted(p.name for p in tmp_path.iterdir())==['input.zip']


def test_robotwin_preparation_finishes_missing_components(tmp_path,monkeypatch):
    root=tmp_path/'release';data=tmp_path/'data';assets=root/'third_party/RoboTwin/assets'
    metadata={'files':{},'extracted_sha256':{}}
    for component in ('background_texture','embodiments','objects'):
        contents=component.encode()
        archive=data/'robotwin-assets'/f'{component}.zip'
        metadata['files'][archive.name]=write_archive(archive,{component+'/probe.bin':contents})
        metadata['extracted_sha256'][component+'/probe.bin']=hashlib.sha256(contents).hexdigest()
    (assets/'embodiments').mkdir(parents=True)
    existing=assets/'embodiments/probe.bin';existing.write_bytes(b'embodiments')
    before=existing.stat().st_mtime_ns
    monkeypatch.setattr(setup_simulators,'ROOT',root)
    monkeypatch.setattr(setup_simulators,'asset_metadata',lambda key:metadata)
    monkeypatch.setattr(setup_simulators,'require_paper_sapien',lambda:None)
    monkeypatch.setattr(setup_simulators,'ROBOTWIN_EDITS',{})
    monkeypatch.setattr(setup_simulators,'run',lambda *args,**kwargs:None)
    setup_simulators.prepare_robotwin(argparse.Namespace(data_root=str(data),dry_run=False))
    for component in ('background_texture','embodiments','objects'):
        assert (assets/component/'probe.bin').read_bytes()==component.encode()
    assert existing.stat().st_mtime_ns==before


def test_failed_clone_does_not_leave_an_unusable_checkout(tmp_path,monkeypatch):
    monkeypatch.setattr(setup_simulators,'ROOT',tmp_path)
    def interrupted(command,dry_run,cwd=None):
        target=Path(command[-1]);target.mkdir();(target/'partial').write_text('partial')
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(setup_simulators,'run',interrupted)
    with pytest.raises(subprocess.CalledProcessError):
        setup_simulators.prepare_libero(argparse.Namespace(benchmark='libero',dry_run=False))
    assert not (tmp_path/'third_party/LIBERO').exists()
    assert not list((tmp_path/'third_party').glob('.jam-clone-*'))


def test_wan_initialization_reports_missing_pinned_assets_before_loading(tmp_path):
    with pytest.raises(SystemExit,match='Download wan and wan-tokenizer') as error:
        init_actions.require_wan_initialization_assets(tmp_path)
    assert 'diffusion_pytorch_model-00003-of-00003.safetensors' in str(error.value)
    assert 'google/umt5-xxl/spiece.model' in str(error.value)
