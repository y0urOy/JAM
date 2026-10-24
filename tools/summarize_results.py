"""Validate both backends' LIBERO result formats before reporting success rates."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from workflow import write_json

ROOT = Path(__file__).resolve().parents[1]
SUITES = {'libero_spatial': 'Spatial', 'libero_object': 'Object',
          'libero_goal': 'Goal', 'libero_10': 'Long'}


def task_list(benchmark):
    if benchmark == 'plus':
        return (ROOT / 'reproduction/libero_plus_paper15_tasks.txt').read_text().splitlines()
    if benchmark == 'libero':
        return [f'{suite},{i}' for suite in ('libero_10', 'libero_goal', 'libero_spatial', 'libero_object')
                for i in range(10)]
    raise ValueError('Expected libero or plus.')


def trials_per_task(benchmark):
    return 1 if benchmark == 'plus' else 50


def summarize(output, benchmark):
    output = Path(output)
    expected = set(task_list(benchmark))
    trials = trials_per_task(benchmark)
    found, errors, episodes = {}, [], []
    for path in sorted(output.rglob('*_results.json')):
        try:
            result = json.loads(path.read_text())
            suite, tid = result['task_suite'], result['task_id']
            if type(tid) is not int:
                raise ValueError('task_id must be an integer')
            key = f'{suite},{tid}'
            if key not in expected:
                raise ValueError(f'unexpected task {key}')
            if key in found:
                raise ValueError(f'duplicate task {key}')
            n, successes = result['total_episodes'], result['successes']
            if type(n) is not int or n != trials:
                raise ValueError(f'expected {trials} episodes, got {n}')
            if type(successes) is not int or not 0 <= successes <= n:
                raise ValueError('invalid successes count')
            good, bad = result['success_episodes'], result['failure_episodes']
            if not isinstance(good, list) or not isinstance(bad, list):
                raise ValueError('episode identities must be lists')
            ids = good + bad
            if any(type(x) is not int for x in ids) or sorted(ids) != list(range(n)):
                raise ValueError('missing, duplicate or out-of-range episode identities')
            if len(good) != successes:
                raise ValueError('success count disagrees with episode identities')
            found[key] = {'task_id': f'{suite}:{tid}', 'suite': suite, 'successes': successes,
                          'episodes': n, 'success_rate_percent': 100 * successes / n,
                          'result_file': str(path.relative_to(output))}
            episodes.extend((f'{suite}:{tid}', episode, int(episode in good)) for episode in range(n))
        except (ValueError, TypeError, KeyError, OSError) as error:
            errors.append(f'{path.relative_to(output)}: {error}')
    missing = sorted(expected - found.keys())
    execution_report = output / 'execution_report.json'
    if execution_report.is_file():
        try:
            execution = json.loads(execution_report.read_text())
            if execution['benchmark'] != benchmark or not execution['workers_completed_without_error']:
                errors.append('Recorded execution failed or belongs to another benchmark; inspect execution_report.json.')
        except (ValueError, TypeError, KeyError, OSError) as error:
            errors.append(f'execution_report.json: {error}')
    complete = not errors and not missing
    valid_successes = sum(x['successes'] for x in found.values())
    valid_episodes = sum(x['episodes'] for x in found.values())
    suite_scores = {}
    for suite, label in SUITES.items():
        rows = [row for row in found.values() if row['suite'] == suite]
        count = sum(row['episodes'] for row in rows)
        suite_scores[label] = (100 * sum(row['successes'] for row in rows) / count) if complete else None
    average = 100 * valid_successes / valid_episodes if complete else None
    suite_scores['AVG'] = average
    return {
        'schema_version': 1, 'benchmark': benchmark, 'complete': complete,
        'expected_tasks': len(expected), 'valid_tasks': len(found),
        'expected_episodes': len(expected) * trials, 'valid_episodes': valid_episodes,
        'valid_successes': valid_successes, 'success_rate_percent': average,
        'scores_percent': suite_scores,
        'aggregation': 'sum(successes) / sum(episodes); task-weighted for the fixed LIBERO-Plus subset',
        'missing_tasks': missing, 'errors': errors,
        'per_task': [found[key] for key in sorted(found)],
    }, sorted(episodes)


def save_summary(output, report, episodes):
    output = Path(output)
    write_json(output / 'summary.json', report)
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Spatial', 'Object', 'Goal', 'Long', 'AVG', 'complete'])
        writer.writerow([*report['scores_percent'].values(), report['complete']])
    with (output / 'episodes.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['task_id', 'episode', 'success'])
        # Do not export a misleading partial sample for paired comparisons.
        if report['complete']:
            writer.writerows(episodes)
