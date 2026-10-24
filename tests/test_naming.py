"""Namespace migration contracts: preserve scientific configs and old run identities."""
import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from compose import compose_recipe
from jam import environment, parser, recipes
from naming import canonical_reference, canonical_recipe_spec
from workflow import load_run

FINGERPRINTS = json.loads((ROOT / 'reproduction/config_fingerprints.json').read_text())


def normalize_config(value, path=''):
    if isinstance(value, dict):
        return {k: normalize_config(v, path + '.' + k) for k, v in value.items()
                if path + '.' + k not in ('.output_dir', '.wandb.name', '.wandb.project')}
    if isinstance(value, list):
        return [normalize_config(x, path) for x in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), '<JAM_SOURCE>')
    return value


@pytest.mark.parametrize('name', FINGERPRINTS['sha256'])
def test_resolved_recipe_matches_recorded_fingerprint(name):
    cfg = compose_recipe(name, '/opt/jam/data', '/opt/jam/models', '/opt/jam/cache')
    normalized = normalize_config(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    actual = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert actual == FINGERPRINTS['sha256'][name]


@pytest.mark.parametrize('name,old_backend,old_task', [
    ('wan-jam', 'fastwam', 'libero_bplus_na_downpred118_lam06_single18_vae_vff4_2cam224_1e-4'),
    ('ovis-jam', 'imagewam', 'libero_ovis_u1_imagewam_foresight'),
])
def test_old_run_recipe_accepts_names_only_and_rejects_experiment_drift(name, old_backend, old_task, tmp_path):
    spec = dict(recipes()[name], backend=old_backend, task=old_task)
    # Use the recorded task spelling from the inverse map where needed.
    from naming import migration
    inverse = {v: k for k, v in migration()['config_names'].items()}
    spec['task'] = inverse.get(recipes()[name]['task'], recipes()[name]['task'])
    record = {'schema_version': 1, 'recipe': name, 'recipe_spec': spec}
    (tmp_path / 'run.json').write_text(json.dumps(record))
    assert load_run(tmp_path, recipes()) == record
    spec['seed'] += 1
    (tmp_path / 'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='recipe changed'):
        load_run(tmp_path, recipes())


def test_legacy_targets_and_asset_ids_are_distinct():
    assert canonical_reference('imagewam.runtime.create_imagewam_ovis_u1') == 'jam_multimodal.runtime.create_jam_ovis_u1'
    assert canonical_reference('fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor') == 'jam_wan.datasets.lerobot.processors.jam_processor.JAMProcessor'
    assert canonical_reference('qwen35_imagewam_native_qwen24_v1') == 'qwen35_imagewam_native_qwen24_v1'
    assert canonical_reference('yuanty/LIBERO-fastwam') == 'yuanty/LIBERO-fastwam'


def test_old_backend_flags_and_environment_keep_explicit_jam_precedence(monkeypatch):
    assert parser().parse_args(['list', '--backend', 'imagewam']).backend == 'multimodal'
    monkeypatch.setenv('IMAGEWAM_VIDEO_BACKEND', 'legacy-reader')
    monkeypatch.delenv('JAM_VIDEO_BACKEND', raising=False)
    assert environment('imagewam', 'libero', {'models': Path('/models')})['JAM_VIDEO_BACKEND'] == 'legacy-reader'
    monkeypatch.setenv('JAM_VIDEO_BACKEND', 'pyav')
    assert environment('multimodal', 'libero', {'models': Path('/models')})['JAM_VIDEO_BACKEND'] == 'pyav'


def test_legacy_checkpoint_formats_remain_distinct_and_exported():
    source = ROOT / 'backends/multimodal/src/jam_multimodal/models/backbones/jam_qwen35.py'
    values = {}
    for node in ast.parse(source.read_text()).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            value = values[node.value.id] if isinstance(node.value, ast.Name) else ast.literal_eval(node.value)
        except (ValueError, KeyError):
            continue
        values[node.targets[0].id] = value
    expected = {'QWEN_FASTWAM_LEGACY_CHECKPOINT_FORMAT': 'qwen35_fastwam_v1',
                'QWEN_FASTWAM_CHECKPOINT_FORMAT': 'qwen35_fastwam_v2',
                'QWEN_IMAGEWAM_LEGACY_CHECKPOINT_FORMAT': 'qwen35_imagewam_v2',
                'QWEN_IMAGEWAM_CHECKPOINT_FORMAT': 'qwen35_imagewam_v3',
                'QWEN_IMAGEWAM_NATIVE_CHECKPOINT_FORMAT': 'qwen35_imagewam_native_qwen24_v1'}
    for key, value in expected.items():
        assert values[key] == value


@pytest.mark.parametrize('backend,package', [('wan', 'jam_wan'), ('multimodal', 'jam_multimodal')])
def test_backend_relative_imports_point_to_released_modules(backend, package):
    src = ROOT / 'backends' / backend / 'src'
    for path in (src / package).rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                pieces = list(path.relative_to(src).parent.parts)
                if node.level > 1:
                    pieces = pieces[:-(node.level - 1)]
                pieces += node.module.split('.') if node.module else []
            elif node.module and node.module.startswith(package + '.'):
                pieces = node.module.split('.')
            else:
                continue
            target = src / Path(*pieces)
            assert target.with_suffix('.py').is_file() or (target / '__init__.py').is_file(), (path, node.lineno, target)
