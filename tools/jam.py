#!/usr/bin/env python3
"""Portable paper-reproduction entrypoints; planning requires only the standard library."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from naming import backend_package, canonical_backend, adopt_legacy_environment
from workflow import (MANAGED_EVAL_KEYS, absolute, begin_training, finish_training, gpu_ids,
                      override_key, python_executable, reject_overrides,
                      resolve_evaluation, run_command)

ROOT = Path(__file__).resolve().parents[1]


def recipes():
    return json.loads((ROOT / 'reproduction/recipes.json').read_text())['recipes']


def value(v):
    return json.dumps(v, separators=(',', ':'))


def assignment(k, v):
    return '++' + k + '=' + value(v)


def paths(args):
    artifacts = Path(os.environ.get('JAM_ARTIFACT_ROOT', ROOT / 'artifacts')).expanduser().resolve()
    return {
        'artifacts': artifacts,
        'data': Path(args.data_root or os.environ.get('JAM_DATA_ROOT', artifacts / 'data')).expanduser().resolve(),
        'models': Path(args.model_root or os.environ.get('JAM_MODEL_ROOT', artifacts / 'models')).expanduser().resolve(),
        'cache': Path(args.cache_root or os.environ.get('JAM_CACHE_ROOT', artifacts / 'cache')).expanduser().resolve(),
    }


def environment(backend, benchmark, p):
    backend = canonical_backend(backend)
    env = adopt_legacy_environment(os.environ.copy(), backend)
    repo = ROOT / 'backends' / backend
    libero = ROOT / 'third_party' / ('LIBERO-plus' if benchmark == 'plus' else 'LIBERO')
    parts = [str(repo / 'src'), str(repo), str(libero), str(repo / 'third_party/flux2/src')]
    if env.get('PYTHONPATH'):
        parts.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(parts)
    env.setdefault('WANDB_MODE', 'disabled')
    env.setdefault('JAM_VIDEO_BACKEND', 'pyav')
    env.setdefault('MUJOCO_GL', 'egl')
    env.setdefault('PYOPENGL_PLATFORM', 'egl')
    env['DIFFSYNTH_MODEL_BASE_PATH'] = str(p['models'])
    # Both readers cache normalization and other run-local files here.
    return env


def data_directories(r, p):
    if r['benchmark'] == 'libero':
        return [str(p['data'] / 'libero_mujoco3.3.2' / (n + '_no_noops_lerobot'))
                for n in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')]
    # The released RoboTwin archive contains a LeRobot root which can itself
    # represent all tasks. Do not select a different robot or split implicitly.
    return [str(p['data'] / 'robotwin2.0/robotwin2.0')]


def base_overrides(r, p, training=True):
    out = ['task=' + r['task'], assignment('data.train.dataset_dirs', data_directories(r, p)),
           assignment('wandb.enabled', False)]
    if r['benchmark'] == 'robotwin':
        out += [assignment('data.val.dataset_dirs', data_directories(r, p)),
                assignment('data.train.pretrained_norm_stats', str(p['data'] / 'robotwin2.0/dataset_stats.json')),
                assignment('data.val.pretrained_norm_stats', str(p['data'] / 'robotwin2.0/dataset_stats.json'))]
        if r['family'] == 'flux':
            out += [assignment('data.val.qwen_text_cache_dir', '${data.train.qwen_text_cache_dir}'),
                    assignment('data.val.qwen_context_len', '${data.train.qwen_context_len}')]
    if r['family'] == 'wan':
        out += [assignment('model.action_dit_pretrained_path', str(p['models'] / 'init/wan_action.pt')),
                assignment('data.train.text_embedding_cache_dir', str(p['cache'] / r['benchmark'] / 'umt5'))]
        if r['benchmark'] == 'robotwin':
            out += [assignment('data.val.text_embedding_cache_dir', str(p['cache'] / r['benchmark'] / 'umt5'))]
    elif r['family'] == 'ovis':
        dim = 14 if r['benchmark'] == 'robotwin' else 7
        out += [assignment('model.ovis_u1_model_path', str(p['models'] / 'Ovis-U1-3B')),
                assignment('model.action_dit_pretrained_path', str(p['models'] / f'init/ovis_action_{dim}d.pt')),
                assignment('data.train.qwen_text_cache_dir', None)]
    elif r['family'] == 'qwen':
        size = r.get('qwen_size', '2B')
        out += [assignment('model.model_id', str(p['models'] / f'Qwen3.5-{size}-Base')),
                assignment('model.teacher_model_id', str(p['models'] / f'Qwen3.5-{size}-Base')),
                assignment('data.train.qwen_text_cache_dir', None)]
    elif r['family'] == 'dino':
        text = str(p['cache'] / r['benchmark'] / 'qwen3_4b_len512')
        out += [assignment('model.ovis_u1_model_path', str(ROOT / 'backends/multimodal/third_party/ovis_u1_hf')),
                assignment('model.ssl_vision.model_id', str(p['models'] / ('dinov2-' + r['dino_size']))),
                assignment('model.ssl_vision.qwen_text_cache_dir', text),
                assignment('data.train.qwen_text_cache_dir', text),
                assignment('data.train.qwen_context_len', 512)]
    elif r['family'] == 'flux':
        dim = 14 if r['benchmark'] == 'robotwin' else 7
        text = str(p['cache'] / r['benchmark'] / 'qwen3_4b_len512')
        out += [assignment('model.flux2_src_path', str(ROOT / 'backends/multimodal/third_party/flux2')),
                assignment('model.flux2_model_path', str(p['models'] / 'flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors')),
                assignment('model.ae_model_path', str(p['models'] / 'flux2/FLUX.2-dev/ae.safetensors')),
                assignment('model.action_dit_pretrained_path', str(p['models'] / f'init/flux_action_{dim}d.pt')),
                assignment('data.train.qwen_text_cache_dir', text),
                assignment('data.train.qwen_context_len', 512),assignment('model.qwen_context_len', 512)]
        if r['benchmark'] == 'robotwin':
            # Preserve the actual released RoboTwin configuration's 128-token
            # cache. LIBERO and DINO use 512. This distinction is documented.
            out += [assignment('data.train.qwen_context_len', 128),assignment('model.qwen_context_len', 128),
                    assignment('data.train.qwen_text_cache_dir', str(p['cache'] / 'robotwin/qwen3_4b_len128'))]
    if training:
        out += [assignment('max_steps', r['steps']),assignment('num_epochs', r['epochs']),
                assignment('seed', r['seed']),assignment('eval_every', 0),
                assignment('persistent_workers', False),assignment('dataloader_multiprocessing_context', 'spawn')]
    out += r['overrides']
    return out


def command_plan(args):
    args = copy.deepcopy(args)
    all_recipes = recipes()
    args.python = python_executable(args.python, args.venv)
    if args.command == 'eval':
        args = resolve_evaluation(args, all_recipes)
    if args.recipe not in all_recipes:
        raise ValueError('Unknown recipe: ' + args.recipe)
    if args.command == 'train-eval':
        return pipeline_plan(args)
    r = all_recipes[args.recipe]
    p = paths(args)
    repo = ROOT / 'backends' / r['backend']
    mode = args.command
    default_out = p['artifacts'] / ('evaluations' if mode == 'eval' else 'runs') / args.recipe
    if mode == 'eval':
        default_out = default_out / args.benchmark / Path(args.checkpoint).stem
    out = absolute(args.output or default_out)
    ov = base_overrides(r, p, training=mode not in ('eval',))
    env = environment(r['backend'], getattr(args, 'benchmark', None) or r['benchmark'], p)
    commands = []
    if mode == 'train':
        reject_overrides(args.override, ('task', 'batch_size', 'gradient_accumulation_steps', 'output_dir'))
        if min(args.nodes, args.gpus_per_node, args.micro_batch) < 1:
            raise ValueError('Nodes, GPUs per node and micro-batch must be positive.')
        if not 0 <= args.node_rank < args.nodes:
            raise ValueError('--node-rank must be in [0, --nodes).')
        if args.grad_accum is not None and args.grad_accum < 1:
            raise ValueError('--grad-accum must be positive.')
        if args.workers < 0:
            raise ValueError('--workers cannot be negative.')
        world = args.nodes * args.gpus_per_node
        micro = args.micro_batch
        accumulation = args.grad_accum or (r['global_batch'] // (world * micro))
        if accumulation < 1 or world * micro * accumulation != r['global_batch']:
            raise ValueError(f'Expected global batch {r["global_batch"]}; world*micro*accum={world}*{micro}*{accumulation}. Set --micro-batch/--grad-accum to match.')
        ov += [assignment('batch_size', micro),assignment('gradient_accumulation_steps', accumulation),
               assignment('num_workers', args.workers),assignment('output_dir', str(out))]
        cmd = [args.python, '-m', 'accelerate.commands.launch', '--config_file',
               str(ROOT / f'environments/accelerate_zero{r["zero_stage"]}.yaml'),
               '--num_processes', str(world), '--num_machines', str(args.nodes),
               '--machine_rank', str(args.node_rank), '--main_process_ip', args.master_addr,
               '--main_process_port', str(args.master_port), 'scripts/train.py']
        commands.append(cmd + ov + args.override)
    elif mode == 'text-cache':
        if r['family'] == 'qwen':
            raise ValueError('Qwen3.5 encodes the instruction in its policy prefix; no separate text cache is needed.')
        if r['family'] in ('flux', 'dino'):
            # DINO's text encoder is Qwen3-4B, loaded by the FLUX cache tool,
            # without constructing a FLUX or DINO visual backbone.
            ov += [assignment('flux2_qwen3_model_spec', str(p['models'] / 'Qwen3-4B'))]
            if r['family'] == 'dino':
                ov += [assignment('model.flux2_src_path', str(ROOT / 'backends/multimodal/third_party/flux2'))]
            script = 'scripts/flux2/precompute_flux2_qwen3_embeds.py'
        elif r['family'] == 'wan':
            script = 'scripts/precompute_text_embeds.py'
        else:
            raise ValueError('Ovis encodes its own condition; no UMT5/Qwen cache is needed.')
        commands.append([args.python, script] + ov + args.override)
    elif mode == 'cache':
        if r['backend'] != 'wan' or 'foresight' not in r['task'] and 'bplus' not in r['task'] and 'vae' not in r['task']:
            raise ValueError('This target-cache entrypoint supports the Wan foresight recipes. Other families train with online frozen anchors.')
        target = str(p['cache'] / args.recipe / 'targets')
        commands.append([args.python,'scripts/extract_foresight_cache.py'] + ov + [
            assignment('output_dir', str(out)),assignment('extract.out_roots',[target]),
            assignment('extract.batch_size', args.micro_batch),assignment('extract.num_workers',args.workers)] + args.override)
    elif mode == 'profile':
        if r['backend'] != 'wan':
            raise ValueError('Use tools/profile_ovis.py for the Ovis timing rows; see docs/ANALYSES.md.')
        commands.append([args.python,'scripts/profile_step_breakdown.py'] + ov + [
            assignment('batch_size', args.micro_batch),assignment('output_dir',str(out))] + args.override)
    elif mode == 'eval':
        args.gpus = ','.join(gpu_ids(args.gpus))
        reject_overrides(args.override, MANAGED_EVAL_KEYS)
        if not args.checkpoint or not args.stats:
            raise ValueError('Evaluation requires both --checkpoint and its matching --stats.')
        benchmark = args.benchmark
        expected = 'robotwin' if benchmark == 'robotwin' else 'libero'
        if r['benchmark'] != expected:
            raise ValueError('Use a matching training recipe; do not silently evaluate a different embodiment.')
        if benchmark == 'robotwin':
            ov += [assignment('ckpt',str(Path(args.checkpoint).expanduser().resolve())),
                   assignment('EVALUATION.dataset_stats_path',str(Path(args.stats).expanduser().resolve())),
                   assignment('EVALUATION.robotwin_root',str(ROOT/'third_party/RoboTwin')),
                   assignment('EVALUATION.output_dir',str(out)),assignment('EVALUATION.exact_output_dir',True),
                   assignment('EVALUATION.instruction_type','unseen'),
                   assignment('EVALUATION.eval_num_episodes',100),assignment('EVALUATION.replan_steps',24),
                   assignment('MULTIRUN.enabled',True),assignment('MULTIRUN.num_gpus',len(args.gpus.split(','))),
                   assignment('MULTIRUN.max_tasks_per_gpu',1),assignment('MULTIRUN.phases',['clean','random'])]
            commands.append([args.python,'experiments/robotwin/run_robotwin_manager.py','--config-name','sim_robotwin'] + ov + args.override)
            env['CUDA_VISIBLE_DEVICES'] = args.gpus
        else:
            commands.append([args.python,str(ROOT/'tools/evaluate_libero.py'),
                '--recipe',args.recipe,'--benchmark',benchmark,'--checkpoint',args.checkpoint,
                '--stats',args.stats,'--gpus',args.gpus,'--output',str(out),
                '--data-root',str(p['data']),'--model-root',str(p['models']),'--cache-root',str(p['cache'])]
                + sum([['--override',x] for x in args.override],[]))
    elif mode == 'compose':
        commands.append([args.python,str(ROOT/'tools/compose.py'),'--recipe',args.recipe,
                         '--data-root',str(p['data']),'--model-root',str(p['models']),
                         '--cache-root',str(p['cache'])] + sum([['--override',x] for x in args.override],[]))
    else:
        raise ValueError(mode)
    plan = {'recipe':args.recipe,'environment':r['environment'],'mode':mode,'cwd':str(repo),
            'output':str(out),'commands':commands,'env':{k:env[k] for k in [
                'PYTHONPATH','WANDB_MODE','JAM_VIDEO_BACKEND','MUJOCO_GL','DIFFSYNTH_MODEL_BASE_PATH',
                'PYOPENGL_PLATFORM','CUDA_VISIBLE_DEVICES'] if k in env},
            'global_batch':r['global_batch'],'scientific_overrides':ov}
    if mode == 'eval':
        plan.update(benchmark=args.benchmark, checkpoint=args.checkpoint, stats=args.stats)
    return plan, env


def pipeline_plan(args):
    """Plan training and matched evaluations without touching model assets."""
    recipe = recipes()[args.recipe]
    train_args = copy.deepcopy(args)
    train_args.command = 'train'
    train, env = command_plan(train_args)
    steps = recipe['steps']
    for raw in args.override:
        if override_key(raw) == 'max_steps':
            try:
                steps = int(raw.split('=', 1)[1])
            except (ValueError, IndexError):
                raise ValueError('train-eval needs an explicit positive integer max_steps.') from None
    if steps < 1:
        raise ValueError('train-eval needs positive max_steps.')
    checkpoint = Path(train['output']) / f'checkpoints/weights/step_{steps:06d}.pt'
    benchmarks = args.benchmarks or [recipe['benchmark']]
    if len(set(benchmarks)) != len(benchmarks):
        raise ValueError('Each evaluation benchmark may appear only once.')
    stages = [train]
    for benchmark in benchmarks:
        evaluation = copy.deepcopy(args)
        evaluation.command = 'eval'
        evaluation.benchmark = benchmark
        evaluation.checkpoint = str(checkpoint)
        evaluation.stats = str(Path(train['output']) / 'dataset_stats.json')
        evaluation.run_dir = None
        evaluation.output = str(Path(train['output']) / 'evaluations' / benchmark / checkpoint.stem)
        evaluation.override = args.override + args.eval_override
        evaluation_plan, _ = command_plan(evaluation)
        stages.append(evaluation_plan)
    return {'mode':'train-eval', 'recipe':args.recipe, 'environment':recipe['environment'],
            'output':train['output'], 'stages':stages,
            'evaluation_node_rank':0, 'expected_checkpoint':str(checkpoint)}, env


def execute_training(args, plan, env):
    record = begin_training(args, plan, recipes()[args.recipe], paths(args), ROOT)
    returncode = 1
    try:
        for command in plan['commands']:
            print(shlex.join(command), flush=True)
            run_command(command, plan['cwd'], env,
                        Path(plan['output']) / f'train_node{args.node_rank}.log')
        returncode = 0
    except subprocess.CalledProcessError as error:
        returncode = error.returncode
        raise
    finally:
        finish_training(plan, record, returncode)


def execute_plan(plan, env):
    if plan['mode'] == 'eval':
        preflight_evaluation(plan)
    for command in plan['commands']:
        print(shlex.join(command), flush=True)
        run_command(command, plan['cwd'], env)


def preflight_evaluation(plan, require_checkpoint=True):
    output = Path(plan['output'])
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty. Use a new --output for a new evaluation.')
    if require_checkpoint:
        for key in ('checkpoint', 'stats'):
            path = Path(plan[key])
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f'Missing or empty evaluation {key}: {path}')
    if plan['benchmark'] in ('libero', 'plus'):
        simulator = ROOT / 'third_party' / ('LIBERO-plus' if plan['benchmark'] == 'plus' else 'LIBERO')
        if not simulator.is_dir():
            raise ValueError(f'Missing simulator {simulator}; see docs/INSTALL.md. No evaluation was started.')


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    listing = sub.add_parser('list', help='List named training/ablation recipes.')
    listing.add_argument('--backend', type=canonical_backend, choices=['wan', 'multimodal'])
    summary = sub.add_parser('summarize', help='Validate and aggregate LIBERO/Plus task results.')
    summary.add_argument('--benchmark', choices=['libero', 'plus'], required=True)
    summary.add_argument('--output', required=True, help='Evaluation directory containing per-task results.')
    for mode in ('train','train-eval','cache','text-cache','profile','eval','compose'):
        p = sub.add_parser(mode)
        p.add_argument('recipe', nargs='?' if mode == 'eval' else None)
        runtime = p.add_mutually_exclusive_group()
        runtime.add_argument('--python', help='Python executable in the policy environment.')
        runtime.add_argument('--venv', help='Policy virtualenv/conda environment directory.')
        p.add_argument('--data-root'); p.add_argument('--model-root'); p.add_argument('--cache-root')
        p.add_argument('--output'); p.add_argument('--override', action='append',default=[])
        p.add_argument('--dry-run',action='store_true')
        p.add_argument('--micro-batch',type=int,default=16)
        p.add_argument('--workers',type=int,default=4)
        if mode in ('train', 'train-eval'):
            p.add_argument('--nodes',type=int,default=1)
            p.add_argument('--gpus-per-node',type=int,default=1)
            p.add_argument('--node-rank',type=int,default=0)
            p.add_argument('--master-addr',default='127.0.0.1')
            p.add_argument('--master-port',type=int,default=29500)
            p.add_argument('--grad-accum',type=int)
        if mode == 'eval':
            p.add_argument('--benchmark',choices=['libero','plus','robotwin'])
            p.add_argument('--run-dir', help='Restore recipe, overrides, latest checkpoint and matching statistics.')
            p.add_argument('--checkpoint');p.add_argument('--stats')
            p.add_argument('--gpus',default='0')
        if mode == 'train-eval':
            p.add_argument('--benchmarks', nargs='+', choices=['libero','plus','robotwin'])
            p.add_argument('--eval-override', action='append', default=[])
            p.add_argument('--gpus',default='0')
    return ap


def main():
    args = parser().parse_args()
    if args.command == 'list':
        for name,r in recipes().items():
            if not args.backend or args.backend == r['backend']:
                print(f'{name:38s} {r["backend"]:8s} env={r["environment"]:5s} {r["benchmark"]:8s} steps={r["steps"]} global_batch={r["global_batch"]}')
        return
    if args.command == 'summarize':
        from summarize_results import summarize, save_summary
        report, episodes = summarize(Path(args.output), args.benchmark)
        save_summary(Path(args.output), report, episodes)
        print(json.dumps(report, indent=2))
        if not report['complete']:
            raise SystemExit(1)
        return
    try:
        plan,env = command_plan(args)
    except ValueError as e:
        raise SystemExit(str(e))
    if args.dry_run:
        print(json.dumps(plan,indent=2));return
    try:
        if args.command == 'train':
            execute_training(args, plan, env)
        elif args.command == 'train-eval':
            for stage in plan['stages'][1:]:
                preflight_evaluation(stage, require_checkpoint=False)
            execute_training(args, plan['stages'][0], env)
            if args.node_rank == 0:
                checkpoint = Path(plan['expected_checkpoint'])
                if not checkpoint.is_file() or not checkpoint.stat().st_size:
                    raise ValueError(f'Training did not produce the final checkpoint: {checkpoint}. Evaluation was not started.')
                for stage in plan['stages'][1:]:
                    stage_env = env.copy()
                    stage_env.update(stage['env'])
                    execute_plan(stage, stage_env)
        else:
            execute_plan(plan, env)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from None


if __name__ == '__main__':
    main()
