"""Namespace migration contracts: preserve scientific configs and old run identities."""
import ast
from functools import lru_cache
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
from naming import canonical_reference, canonical_recipe_spec, migration
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
    ('wan-jam', 'fastwam', 'libero_bplus_na_downpred118_lam06_s18_vae_off12_2cam224_1e-4'),
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


BACKEND_PACKAGES = {'wan': ('jam_wan', 'fastwam'), 'multimodal': ('jam_multimodal', 'imagewam')}


@lru_cache(maxsize=None)
def top_level_names(path):
    """Names a module defines or re-exports at import time, and whether it resolves names lazily."""
    names, lazy = set(), False
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            lazy |= node.name == '__getattr__'
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                lazy |= alias.name == '*'
                names.add((alias.asname or alias.name).split('.')[0])
    return names, lazy


@pytest.mark.parametrize('backend', BACKEND_PACKAGES)
def test_backend_imports_point_to_released_modules(backend):
    base = ROOT / 'backends' / backend
    src = base / 'src'
    for path in [p for folder in ('src', 'scripts', 'experiments') for p in (base / folder).rglob('*.py')]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports = [(alias.name.split('.'), []) for alias in node.names
                           if alias.name.split('.')[0] in BACKEND_PACKAGES[backend]]
            elif isinstance(node, ast.ImportFrom) and node.level and path.is_relative_to(src):
                pieces = list(path.relative_to(src).parent.parts)
                if node.level > 1:
                    pieces = pieces[:-(node.level - 1)]
                imports = [(pieces + (node.module.split('.') if node.module else []), node.names)]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in BACKEND_PACKAGES[backend]:
                imports = [(node.module.split('.'), node.names)]
            else:
                continue
            for pieces, aliases in imports:
                canonical, legacy = BACKEND_PACKAGES[backend]
                pieces = [canonical if piece == legacy and not index else piece for index, piece in enumerate(pieces)]
                target = src / Path(*pieces)
                module = target.with_suffix('.py') if target.with_suffix('.py').is_file() else target / '__init__.py'
                assert module.is_file(), (path, node.lineno, target)
                names, lazy = top_level_names(module)
                for alias in aliases:
                    if lazy or alias.name == '*' or alias.name in names or alias.name.startswith('__'):
                        continue
                    submodule = target / alias.name
                    assert submodule.with_suffix('.py').is_file() or (submodule / '__init__.py').is_file(), (
                        path, node.lineno, '.'.join(pieces), alias.name)


def test_migration_map_points_to_released_files():
    mapping = migration()
    for old, new in mapping['paths'].items():
        assert (ROOT / new).is_file(), (old, new)
    sources = [ROOT / 'backends' / backend / 'src' for backend in BACKEND_PACKAGES]
    defined = set()
    for old, new in mapping['modules'].items():
        candidates = [src / Path(*new.split('.')) for src in sources]
        modules = [c.with_suffix('.py') for c in candidates if c.with_suffix('.py').is_file()]
        modules += [c / '__init__.py' for c in candidates if (c / '__init__.py').is_file()]
        assert modules, (old, new)
        defined.update(top_level_names(modules[0])[0])
    for old, new in mapping['symbols'].items():
        assert new in defined, (old, new)
    configs = {path.stem for backend in BACKEND_PACKAGES
               for path in (ROOT / 'backends' / backend / 'configs').rglob('*.yaml')}
    for old, new in mapping['config_names'].items():
        assert new in configs, (old, new)
    for path in mapping['excluded_incomplete_prototypes']['files']:
        assert not (ROOT / path).exists(), path
