#!/usr/bin/env python3
"""Shared scalar LIBERO/Plus execution and result validation for both backends."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from jam import ROOT, recipes, paths, base_overrides, environment, assignment
from summarize_results import task_list, trials_per_task, summarize, save_summary
from workflow import MANAGED_EVAL_KEYS, absolute, gpu_ids, reject_overrides, write_json


def build_plan(args):
    recipe = recipes()[args.recipe]
    p = paths(args)
    if recipe['benchmark'] != 'libero':
        raise ValueError('A LIBERO training recipe is required.')
    reject_overrides(args.override, MANAGED_EVAL_KEYS)
    gpus = gpu_ids(args.gpus)
    tasks = task_list(args.benchmark)
    output = absolute(args.output)
    overrides = base_overrides(recipe, p, training=False) + [
        assignment('EVALUATION.replan_steps', 8 if recipe['backend'] == 'multimodal' else 10),
    ] + args.override + [
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
    else:
        overrides += [assignment('model.action_dit_pretrained_path', None)]
    workers = []
    for index, gpu in enumerate(gpus):
        shard = tasks[index::len(gpus)]
        if not shard:
            continue
        key = 'EVALUATION.task_file' if recipe['backend'] == 'wan' else 'EVALUATION.task_chunk_file'
        suite, tid = shard[0].split(',')
        task_file = output / f'tasks_gpu{index}.txt'
        command = [sys.executable, 'experiments/libero/eval_libero_single.py', '--config-name',
                   'sim_libero_plus' if args.benchmark == 'plus' else 'sim_libero'] + overrides + [
                       assignment(key, str(task_file)),
                       assignment('EVALUATION.task_suite_name', suite),
                       assignment('EVALUATION.task_id', int(tid))]
        workers.append({'gpu': gpu, 'tasks': shard, 'command': command,
                        'task_file': str(task_file), 'log': f'worker_{index}.log'})
    return {
        'recipe': args.recipe, 'benchmark': args.benchmark, 'backend': recipe['backend'],
        'checkpoint': str(absolute(args.checkpoint)), 'stats': str(absolute(args.stats)),
        'task_count': len(tasks), 'trials_per_task': trials_per_task(args.benchmark),
        'scalar_policy_inference': True, 'overrides': args.override,
        'output': str(output), 'workers': workers,
    }


def execute(args, plan):
    for path in (plan['checkpoint'], plan['stats']):
        if not Path(path).is_file() or not Path(path).stat().st_size:
            raise ValueError('Missing or empty evaluation asset: ' + path)
    simulator = ROOT / 'third_party' / ('LIBERO-plus' if args.benchmark == 'plus' else 'LIBERO')
    if not simulator.exists():
        raise ValueError('Install the pinned simulator checkout; see docs/INSTALL.md.')
    output = Path(plan['output'])
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty. Use a new --output to avoid mixing checkpoints or stale task results.')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'evaluation_plan.json', plan)
    p = paths(args)
    repo = ROOT / 'backends' / plan['backend']

    def run(item):
        Path(item['task_file']).write_text('\n'.join(item['tasks']) + '\n')
        env = environment(plan['backend'], args.benchmark, p)
        env['CUDA_VISIBLE_DEVICES'] = item['gpu']
        with (output / item['log']).open('w') as log:
            try:
                code = subprocess.run(item['command'], cwd=repo, env=env,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            except OSError as error:
                log.write(str(error) + '\n')
                code = 1
        return {'gpu': item['gpu'], 'assigned_task_count': len(item['tasks']),
                'exit_code': code, 'log': item['log']}

    with ThreadPoolExecutor(max_workers=len(plan['workers'])) as pool:
        results = list(pool.map(run, plan['workers']))
    workers_ok = all(x['exit_code'] == 0 for x in results)
    summary, episodes = summarize(output, args.benchmark)
    if not workers_ok:
        summary['complete'] = False
        summary['success_rate_percent'] = None
        summary['scores_percent'] = dict.fromkeys(summary['scores_percent'])
        summary['errors'].append('One or more evaluator processes failed; inspect worker logs.')
    summary['recipe'] = args.recipe
    summary['checkpoint'] = plan['checkpoint']
    summary['overrides'] = args.override
    save_summary(output, summary, episodes)
    report = {'recipe': args.recipe, 'benchmark': args.benchmark,
              'task_count': plan['task_count'], 'workers_completed_without_error': workers_ok,
              'results_complete': summary['complete'], 'workers': results,
              'success_rate_percent': summary['success_rate_percent'], 'summary': 'summary.json'}
    write_json(output / 'execution_report.json', report)
    print(json.dumps(report, indent=2))
    return 0 if summary['complete'] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--benchmark', choices=['libero', 'plus'], required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--stats', required=True)
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--output', required=True)
    ap.add_argument('--data-root'); ap.add_argument('--model-root'); ap.add_argument('--cache-root')
    ap.add_argument('--override', action='append', default=[])
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
