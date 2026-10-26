#!/usr/bin/env python3
"""Shared scalar LIBERO/Plus execution and result validation for both backends."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event, Lock
import time

from jam import ROOT, recipes, paths, base_overrides, environment, assignment, libero_config_dir
from naming import canonical_recipe
from summarize_results import task_list, trials_per_task, read_results, summarize, save_summary
from workflow import MANAGED_EVAL_KEYS, absolute, gpu_ids, override_key, reject_overrides, write_json

# A crashed or stalled simulator (native EGL failures occur on some GPU nodes) costs only its unfinished
# tasks: they run again in fresh workers.
RETRIES = 2
STALL_SECONDS = 7200
POLL_SECONDS = 30
RESUME_KEYS = ('recipe', 'benchmark', 'checkpoint', 'stats', 'overrides', 'task_count', 'trials_per_task')


def resume_identity(plan):
    # Keep the resolved model/data/cache paths and scientific overrides, not only
    # the user's raw --override list. GPU assignment and the interpreter can change
    # when rescheduling the same evaluation; output and task shards are operational.
    operational = {'gpu_id', 'EVALUATION.output_dir', 'EVALUATION.task_file', 'EVALUATION.task_chunk_file'}
    command = [part for part in plan.get('command', [])[1:] if override_key(part) not in operational]
    return {**{key: canonical_recipe(plan.get(key)) if key == 'recipe' else plan.get(key) for key in RESUME_KEYS},
            'backend': plan.get('backend'), 'resolved_command': command}


def build_plan(args):
    name = canonical_recipe(args.recipe)
    recipe = recipes()[name]
    p = paths(args)
    if recipe['benchmark'] != 'libero':
        raise ValueError('A LIBERO training recipe is required.')
    reject_overrides(args.override, MANAGED_EVAL_KEYS)
    gpus = gpu_ids(args.gpus)
    tasks = task_list(args.benchmark)
    output = absolute(args.output)
    # Replanning interval of the recorded evaluations: Wan and Qwen3.5 every 10 steps; Ovis-U1, FLUX.2
    # and DINOv2 every 8, except DINOv2 on LIBERO-Plus (10).
    replan = 10 if (recipe['backend'] == 'wan' or recipe['family'] == 'qwen'
                    or (recipe['family'] == 'dino' and args.benchmark == 'plus')) else 8
    recorded = [assignment('EVALUATION.replan_steps', replan),
                assignment('model.action_dit_pretrained_path', None)]  # the checkpoint holds the action expert
    if recipe['family'] == 'qwen':
        # The native Qwen action expert is built from the source Qwen weights; the recorded
        # evaluation used the backend processor and computed every prefix row.
        recorded += [assignment('model.skip_dit_load_from_pretrain', False),
                     assignment('model.tensor_native_preprocessing', False),
                     assignment('model.action_dit_config.skip_discarded_prefix_mlp', False),
                     assignment('model.action_dit_config.skip_discarded_prefix_query', False)]
    if recipe['family'] == 'flux':  # online instruction encoding with the pinned Qwen3-4B
        recorded += [assignment('model.qwen3_model_spec', str(p['models'] / 'Qwen3-4B'))]
    overrides = base_overrides(recipe, p, training=False) + recorded + args.override + [
        assignment('ckpt', str(absolute(args.checkpoint))),
        assignment('EVALUATION.dataset_stats_path', str(absolute(args.stats))),
        assignment('EVALUATION.num_trials', trials_per_task(args.benchmark)),
        assignment('EVALUATION.disable_foresight', True),
        assignment('EVALUATION.env_subprocess', True),
        assignment('EVALUATION.env_num', 1),
        assignment('EVALUATION.output_dir', str(output)),
        assignment('EVALUATION.canonical_result_names', True),
        assignment('gpu_id', 0),
    ]
    if recipe['backend'] == 'multimodal':
        overrides += [assignment('EVALUATION.strict_scalar_inference', True),
                      assignment('EVALUATION.exact_output_dir', True),
                      assignment('EVALUATION.atomic_result_writes', True)]
    plan = {
        'recipe': name, 'benchmark': args.benchmark, 'backend': recipe['backend'],
        'checkpoint': str(absolute(args.checkpoint)), 'stats': str(absolute(args.stats)),
        'task_count': len(tasks), 'trials_per_task': trials_per_task(args.benchmark),
        'scalar_policy_inference': True, 'overrides': args.override,
        'output': str(output), 'gpus': gpus,
        'command': [sys.executable, 'experiments/libero/eval_libero_single.py', '--config-name',
                    'sim_libero_plus' if args.benchmark == 'plus' else 'sim_libero'] + overrides,
        'task_key': 'EVALUATION.task_file' if recipe['backend'] == 'wan' else 'EVALUATION.task_chunk_file',
    }
    plan['workers'] = shard_workers(plan, tasks)
    return plan


def shard_workers(plan, tasks, attempt=0):
    """One worker per GPU over an interleaved share of the tasks."""
    output = Path(plan['output'])
    suffix = f'_retry{attempt}' if attempt else ''
    workers = []
    for index, gpu in enumerate(plan['gpus']):
        shard = tasks[index::len(plan['gpus'])]
        if not shard:
            continue
        suite, tid = shard[0].split(',')
        task_file = output / f'tasks_gpu{index}{suffix}.txt'
        command = plan['command'] + [assignment(plan['task_key'], str(task_file)),
                                     assignment('EVALUATION.task_suite_name', suite),
                                     assignment('EVALUATION.task_id', int(tid))]
        workers.append({'gpu': gpu, 'tasks': shard, 'command': command,
                        'task_file': str(task_file), 'log': f'worker_{index}{suffix}.log'})
    return workers


def unfinished_tasks(output, benchmark):
    """Tasks without a valid result; result files that fail validation are set aside as *.invalid."""
    found, invalid, _ = read_results(output, benchmark)
    for path, _ in invalid:
        destination = path.with_name(path.name + '.invalid')
        suffix = 1
        while destination.exists():
            destination = path.with_name(path.name + f'.invalid.{suffix}')
            suffix += 1
        path.rename(destination)
    return [task for task in task_list(benchmark) if task not in found]


def execute(args, plan):
    for path in (plan['checkpoint'], plan['stats']):
        if not Path(path).is_file() or not Path(path).stat().st_size:
            raise ValueError('Missing or empty evaluation asset: ' + path)
    simulator = ROOT / 'third_party' / ('LIBERO-plus' if args.benchmark == 'plus' else 'LIBERO')
    if not (libero_config_dir(args.benchmark, ROOT) / 'config.yaml').is_file() \
            or not (simulator / 'libero/libero/assets').is_dir():
        raise ValueError('Prepare the pinned simulator with python tools/setup_simulators.py '
                         + ('plus' if args.benchmark == 'plus' else 'libero') + '; see docs/INSTALL.md.')
    output = Path(plan['output'])
    if output.exists() and any(output.iterdir()):
        if not getattr(args, 'resume', False):
            raise ValueError(f'{output} is not empty. Use a new --output, or --resume to finish the same evaluation.')
        try:
            recorded = json.loads((output / 'evaluation_plan.json').read_text())
        except (OSError, ValueError):
            raise ValueError(f'{output} has no readable evaluation_plan.json to resume.') from None
        if resume_identity(recorded) != resume_identity(plan):
            raise ValueError(f'{output} holds a different evaluation. Use a new --output.')
        (output / 'execution_report.json').unlink(missing_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'evaluation_plan.json', plan)
    p = paths(args)
    repo = ROOT / 'backends' / plan['backend']
    cancelled = Event()
    active, active_lock = set(), Lock()

    def stop_process(process):
        # A policy may crash while its simulator child remains alive. The process
        # group still exists after its leader exits, so always clean it up.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def cancel_workers():
        cancelled.set()
        with active_lock:
            processes = list(active)
        for process in processes:
            stop_process(process)

    def run(item):
        Path(item['task_file']).write_text('\n'.join(item['tasks']) + '\n')
        env = environment(plan['backend'], args.benchmark, p)
        env['CUDA_VISIBLE_DEVICES'] = item['gpu']
        targets = [output / suite / f'task{tid}_results.json' for suite, tid in (x.split(',') for x in item['tasks'])]
        with (output / item['log']).open('w') as log:
            process = None
            try:
                if cancelled.is_set():
                    return {'gpu': item['gpu'], 'assigned_task_count': len(item['tasks']),
                            'exit_code': 1, 'log': item['log']}
                try:
                    process = subprocess.Popen(item['command'], cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as error:
                    log.write(str(error) + '\n')
                    return {'gpu': item['gpu'], 'assigned_task_count': len(item['tasks']), 'exit_code': 1, 'log': item['log']}
                with active_lock:
                    active.add(process)
                finished, progress = -1, time.monotonic()
                while process.poll() is None:
                    if cancelled.wait(POLL_SECONDS):
                        break
                    count = sum(path.is_file() for path in targets)
                    if count != finished:
                        finished, progress = count, time.monotonic()
                    elif time.monotonic() - progress > STALL_SECONDS:
                        log.write(f'\nNo task finished for {STALL_SECONDS} s; stopping this worker.\n')
                        log.flush()
                        break
            finally:
                if process is not None:
                    try:
                        stop_process(process)
                    finally:
                        with active_lock:
                            active.discard(process)
        return {'gpu': item['gpu'], 'assigned_task_count': len(item['tasks']),
                'exit_code': process.returncode, 'log': item['log']}

    attempts = []
    for attempt in range(RETRIES + 1):
        pending = unfinished_tasks(output, args.benchmark)
        if not pending:
            break
        workers = shard_workers(plan, pending, attempt)
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            try:
                attempts.append(list(pool.map(run, workers)))
            except BaseException:
                # Cancel before the executor waits for its threads, including on Ctrl-C.
                cancel_workers()
                raise
    results = [worker for attempt in attempts for worker in attempt]
    summary, episodes = summarize(output, args.benchmark)
    workers_ok = summary['complete']
    if not workers_ok:
        summary['errors'].append(f'Tasks without a valid result after {len(attempts)} attempts; inspect the worker logs.')
    summary['recipe'] = plan['recipe']
    summary['checkpoint'] = plan['checkpoint']
    summary['overrides'] = args.override
    save_summary(output, summary, episodes)
    report = {'recipe': plan['recipe'], 'benchmark': args.benchmark,
              'task_count': plan['task_count'], 'workers_completed_without_error': workers_ok,
              'results_complete': summary['complete'], 'attempts': len(attempts), 'workers': results,
              'success_rate_percent': summary['success_rate_percent'], 'summary': 'summary.json'}
    write_json(output / 'execution_report.json', report)
    print(json.dumps(report, indent=2))
    return 0 if summary['complete'] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', required=True, type=canonical_recipe)
    ap.add_argument('--benchmark', choices=['libero', 'plus'], required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--stats', required=True)
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--output', required=True)
    ap.add_argument('--data-root'); ap.add_argument('--model-root'); ap.add_argument('--cache-root')
    ap.add_argument('--override', action='append', default=[])
    ap.add_argument('--resume', action='store_true', help='Finish an interrupted evaluation in the same --output.')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return
        raise SystemExit(execute(args, plan))
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None


if __name__ == '__main__':
    main()
