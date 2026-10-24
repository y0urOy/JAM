"""Shared run artifacts and process handling for both policy backends.

Planning and artifact discovery use only the standard library, without importing
either model or probing CUDA. A run directory can be moved as a single unit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from naming import canonical_recipe_spec, canonical_reference

MANAGED_EVAL_KEYS = (
    'task', 'ckpt', 'gpu_id', 'output_dir', 'hydra', 'MULTIRUN',
    'EVALUATION.task_file', 'EVALUATION.task_chunk_file', 'EVALUATION.task_suite_name',
    'EVALUATION.task_id', 'EVALUATION.num_trials', 'EVALUATION.output_dir',
    'EVALUATION.dataset_stats_path', 'EVALUATION.exact_output_dir',
    'EVALUATION.canonical_result_names', 'EVALUATION.atomic_result_writes',
    'EVALUATION.env_num', 'EVALUATION.strict_scalar_inference',
    'EVALUATION.task_name', 'EVALUATION.task_config', 'EVALUATION.eval_num_episodes',
)


def absolute(path):
    return Path(path).expanduser().resolve()


def python_executable(python=None, venv=None):
    # Do not resolve the bin/python symlink: that would bypass its virtualenv.
    if venv:
        return str(absolute(venv) / 'bin/python')
    candidate = python or sys.executable
    if os.sep in candidate:
        return os.path.abspath(os.path.expanduser(candidate))
    return shutil.which(candidate) or candidate


def gpu_ids(raw):
    ids = [part.strip() for part in raw.split(',')]
    if not ids or any(not part for part in ids) or len(set(ids)) != len(ids):
        raise ValueError('Provide distinct, nonempty GPU IDs, for example --gpus 0,1.')
    return ids


def override_key(raw):
    return raw.split('=', 1)[0].lstrip('+~')


def reject_overrides(overrides, keys):
    for raw in overrides:
        key = override_key(raw)
        if any(key == item or key.startswith(item + '.') or item.startswith(key + '.') for item in keys):
            raise ValueError(f'{key} is managed by the unified launcher; use its named options instead.')


def latest_checkpoint(run_dir):
    candidates = []
    for path in (absolute(run_dir) / 'checkpoints/weights').glob('step_*.pt'):
        match = re.fullmatch(r'step_(\d+)\.pt', path.name)
        if match and path.is_file() and path.stat().st_size:
            candidates.append((int(match[1]), path))
    if not candidates:
        raise ValueError(f'No nonempty step_*.pt checkpoint in {run_dir}/checkpoints/weights.')
    return max(candidates, key=lambda item: item[0])[1]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_version(root):
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True))
        return {'commit': commit, 'tracked_changes': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'tracked_changes': None}


def load_run(run_dir, all_recipes):
    root = absolute(run_dir)
    path = root / 'run.json'
    if not path.is_file():
        raise ValueError(f'Missing {path}; for an older run, supply the recipe, --checkpoint and --stats explicitly.')
    record = json.loads(path.read_text())
    name = record.get('recipe')
    if record.get('schema_version') != 1 or name not in all_recipes:
        raise ValueError(f'Unsupported run metadata: {path}')
    if canonical_recipe_spec(record.get('recipe_spec')) != all_recipes[name]:
        raise ValueError('The recipe changed since training. Use the recorded source version or explicitly evaluate the original config.')
    for filename, expected in record.get('artifact_sha256', {}).items():
        if filename not in ('config.yaml', 'dataset_stats.json'):
            raise ValueError('Unexpected run artifact in metadata.')
        if not (root / filename).is_file() or file_sha256(root / filename) != expected:
            raise ValueError(f'Training artifact changed or missing: {root / filename}')
    return record


def resolve_evaluation(args, all_recipes):
    """Restore one training run's recipe, overrides and paired statistics."""
    if args.run_dir:
        root = absolute(args.run_dir)
        record = load_run(root, all_recipes)
        if args.recipe and args.recipe != record['recipe']:
            raise ValueError('--recipe must match the recipe saved in --run-dir.')
        if args.stats:
            raise ValueError('--run-dir always uses its own dataset_stats.json; omit --stats.')
        args.recipe = record['recipe']
        for key in ('data', 'model', 'cache'):
            attr = key + '_root'
            if not getattr(args, attr) and not os.environ.get('JAM_' + key.upper() + '_ROOT'):
                setattr(args, attr, record['paths'][key])
        args.override = [canonical_reference(x) for x in record['training_overrides']] + args.override
        args.stats = str(root / 'dataset_stats.json')
        if args.checkpoint:
            checkpoint = absolute(args.checkpoint)
            try:
                checkpoint.relative_to(root / 'checkpoints/weights')
            except ValueError:
                raise ValueError('--checkpoint must belong to --run-dir/checkpoints/weights.') from None
        else:
            checkpoint = latest_checkpoint(root)
        args.checkpoint = str(checkpoint)
    if not args.recipe or args.recipe not in all_recipes:
        raise ValueError('Supply a known recipe or --run-dir from unified training.')
    if not args.checkpoint or not args.stats:
        raise ValueError('Evaluation requires --run-dir or both --checkpoint and matching --stats.')
    args.checkpoint = str(absolute(args.checkpoint))
    args.stats = str(absolute(args.stats))
    args.benchmark = args.benchmark or all_recipes[args.recipe]['benchmark']
    if args.run_dir and not args.output:
        args.output = str(absolute(args.run_dir) / 'evaluations' / args.benchmark / Path(args.checkpoint).stem)
    return args


def begin_training(args, plan, recipe, path_roots, root):
    out = Path(plan['output'])
    # Rank zero owns the shared metadata. Other nodes only write their own log.
    if args.node_rank != 0:
        return None
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.yaml').exists() or any((out / 'checkpoints/weights').glob('*.pt')):
        raise ValueError(f'{out} already contains training artifacts. Choose a new --output; use backend scripts for explicit state resumption.')
    record = {
        'schema_version': 1, 'recipe': args.recipe, 'recipe_spec': recipe,
        'status': 'training', 'started_at': datetime.now(timezone.utc).isoformat(),
        'source': source_version(root), 'environment': recipe['environment'],
        'paths': {key: str(path_roots[name]) for key, name in (
            ('data', 'data'), ('model', 'models'), ('cache', 'cache'))},
        'training_overrides': args.override, 'plan': plan,
        'artifact_sha256': {},
    }
    try:
        with (out / 'run.json').open('x') as handle:
            json.dump(record, handle, indent=2)
            handle.write('\n')
    except FileExistsError:
        raise ValueError(f'{out}/run.json exists. Choose a new --output to preserve the earlier run.') from None
    return record


def finish_training(plan, record, returncode):
    if record is None:
        return
    out = Path(plan['output'])
    record['status'] = 'completed' if returncode == 0 else 'failed'
    record['returncode'] = returncode
    record['finished_at'] = datetime.now(timezone.utc).isoformat()
    for filename in ('config.yaml', 'dataset_stats.json'):
        path = out / filename
        if path.is_file():
            record['artifact_sha256'][filename] = file_sha256(path)
    write_json(out / 'run.json', record)


def run_command(command, cwd, env, log_path=None):
    """Stream training output to the console and a per-node log."""
    if log_path is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('x') as log:
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            try:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code, command)
