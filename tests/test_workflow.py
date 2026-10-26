"""CPU integration tests: real command planning, fake neural/simulator workers."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from threading import Lock
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import jam
import evaluate_libero
from summarize_results import summarize, save_summary, task_list, trials_per_task
from workflow import (begin_training, finish_training, latest_checkpoint, load_run,
                      python_executable)


LIBERO_RECIPES = ['wan-jam', 'ovis-jam', 'qwen-jam', 'dino-b-jam', 'dino-l-jam', 'flux-jam-libero']
ROBOTWIN_RECIPES = ['robotwin-wan-jam', 'robotwin-ovis-jam', 'robotwin-flux-jam', 'robotwin-qwen-jam']


def args(*words):
    return jam.parser().parse_args(list(words))


@pytest.mark.parametrize('recipe', LIBERO_RECIPES)
def test_pipeline_routes_same_checkpoint_and_config(recipe, tmp_path):
    parsed = args('train-eval', recipe, '--output', str(tmp_path / 'run with spaces'),
                  '--venv', str(tmp_path / 'policy env'), '--benchmarks', 'libero', 'plus',
                  '--override', 'model.action_dit_config.use_gradient_checkpointing=true',
                  '--gpus', '2,5', '--gpus-per-node', '8')
    plan, _ = jam.command_plan(parsed)
    assert len(plan['stages']) == 3
    train, libero, plus = plan['stages']
    assert train['mode'] == 'train'
    assert libero['output'] != plus['output']
    assert all(stage['commands'][0][0] == str(tmp_path / 'policy env/bin/python') for stage in plan['stages'])
    for stage in (libero, plus):
        command = stage['commands'][0]
        assert command[command.index('--checkpoint') + 1] == plan['expected_checkpoint']
        assert command[command.index('--stats') + 1] == str(tmp_path / 'run with spaces/dataset_stats.json')
        assert 'model.action_dit_config.use_gradient_checkpointing=true' in command
    assert not list(tmp_path.iterdir())  # planning never creates runs or loads weights


@pytest.mark.parametrize('recipe', ROBOTWIN_RECIPES)
def test_robotwin_pipeline_evaluates_its_own_checkpoint(recipe, tmp_path):
    plan, _ = jam.command_plan(args('train-eval', recipe, '--output', str(tmp_path / 'run'),
                                    '--gpus-per-node', '8', '--gpus', '0,1'))
    train, robotwin = plan['stages']
    assert train['mode'] == 'train' and robotwin['benchmark'] == 'robotwin'
    command = robotwin['commands'][0]
    assert command[1:4] == ['experiments/robotwin/run_robotwin_manager.py', '--config-name', 'sim_robotwin']
    assert '++ckpt=' + json.dumps(plan['expected_checkpoint']) in command
    stats = str(Path(train['output']) / 'dataset_stats.json')
    assert '++EVALUATION.dataset_stats_path=' + json.dumps(stats) in command
    assert '++MULTIRUN.num_gpus=2' in command and robotwin['env']['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert '++model.action_dit_pretrained_path=null' in command
    multimodal = recipe != 'robotwin-wan-jam'
    assert robotwin['env'].get('JAM_FORESIGHT_NO_TEACHER') == ('1' if multimodal else None)
    nonidle = [x for x in train['commands'][0] if x.startswith('++data.train.nonidle_filter_path=')]
    assert len(nonidle) == int(multimodal)
    assert all(x.endswith('robotwin2.0/robotwin2.0/nonidle_ranges.json"') for x in nonidle)
    assert robotwin['output'] == str(Path(train['output']) / 'evaluations/robotwin' / Path(plan['expected_checkpoint']).stem)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('options', [('--nodes','0'), ('--micro-batch','0'), ('--grad-accum','0'),
                                     ('--node-rank','1'), ('--workers','-1')])
def test_invalid_topology_rejected(options):
    with pytest.raises(ValueError):
        jam.command_plan(args('train', 'wan-jam', *options))


@pytest.mark.parametrize('override', ['batch_size=1', '++gradient_accumulation_steps=7',
                                     '++output_dir=/another/run', 'task=some_other_model'])
def test_training_cannot_silently_bypass_topology_and_run_directory(override):
    with pytest.raises(ValueError, match='managed'):
        jam.command_plan(args('train', 'wan-jam', '--override', override))


def test_explicit_step_and_pipeline_protocol_validation(tmp_path):
    plan, _ = jam.command_plan(args('train-eval', 'wan-jam', '--override', 'max_steps=3'))
    assert plan['expected_checkpoint'].endswith('step_000003.pt')
    for benchmarks in [('robotwin',), ('libero', 'libero')]:
        with pytest.raises(ValueError):
            jam.command_plan(args('train-eval', 'wan-jam', '--benchmarks', *benchmarks))


def make_run(tmp_path, recipe='ovis-jam'):
    run = tmp_path / 'original'
    parsed = args('train', recipe, '--output', str(run), '--override', 'seed=7',
                  '--override', 'model.foresight.lambda_foresight=0.25')
    plan, _ = jam.command_plan(parsed)
    record = begin_training(parsed, plan, jam.recipes()[recipe], jam.paths(parsed), ROOT)
    (run / 'config.yaml').write_text('seed: 7\n')
    (run / 'dataset_stats.json').write_text('{}\n')
    weights = run / 'checkpoints/weights'
    weights.mkdir(parents=True)
    for name in ('step_9.pt', 'step_10.pt'):
        (weights / name).write_bytes(b'explicit test double, not a checkpoint')
    (weights / 'step_999.pt').touch()  # interrupted empty write must not win
    finish_training(plan, record, 0)
    return run


def test_relocated_run_preserves_pairing_and_overrides(tmp_path):
    run = make_run(tmp_path)
    moved = tmp_path / 'moved run'
    run.rename(moved)
    assert latest_checkpoint(moved).name == 'step_10.pt'
    parsed = args('eval', '--run-dir', str(moved), '--benchmark', 'plus',
                  '--model-root', str(tmp_path / 'relocated models'))
    plan, _ = jam.command_plan(parsed)
    command = plan['commands'][0]
    assert plan['recipe'] == 'ovis-jam'
    assert str(moved / 'checkpoints/weights/step_10.pt') in command
    assert str(moved / 'dataset_stats.json') in command
    assert 'seed=7' in command
    assert 'model.foresight.lambda_foresight=0.25' in command
    assert str(tmp_path / 'relocated models') in command
    assert plan['output'] == str(moved / 'evaluations/plus/step_10')


def test_run_recorded_with_an_environment_profile_still_evaluates(tmp_path):
    run = make_run(tmp_path)
    record = json.loads((run / 'run.json').read_text())
    assert 'environment' not in record and 'environment' not in record['recipe_spec']
    record['environment'] = record['recipe_spec']['environment'] = 'ovis'
    (run / 'run.json').write_text(json.dumps(record))
    plan, _ = jam.command_plan(args('eval', '--run-dir', str(run), '--benchmark', 'libero'))
    assert plan['recipe'] == 'ovis-jam' and 'environment' not in plan
    assert str(run / 'checkpoints/weights/step_10.pt') in plan['commands'][0]


def test_run_rejects_wrong_model_stats_and_checkpoint(tmp_path):
    run = make_run(tmp_path)
    for tail in [('wan-jam',), ('--stats', '/wrong/stats.json'), ('--checkpoint', '/wrong/step_10.pt')]:
        with pytest.raises(ValueError):
            jam.command_plan(args('eval', '--run-dir', str(run), *tail))
    (run / 'dataset_stats.json').write_text('{"different":true}')
    with pytest.raises(ValueError, match='artifact changed'):
        jam.command_plan(args('eval', '--run-dir', str(run)))


def test_recipe_drift_and_existing_run_are_rejected(tmp_path):
    run = make_run(tmp_path)
    changed = copy.deepcopy(jam.recipes())
    changed['ovis-jam']['overrides'].append('model.foresight.offsets=[0]')
    with pytest.raises(ValueError, match='recipe changed'):
        load_run(run, changed)
    parsed = args('train', 'ovis-jam', '--output', str(run))
    plan, _ = jam.command_plan(parsed)
    with pytest.raises(ValueError, match='already contains'):
        begin_training(parsed, plan, jam.recipes()['ovis-jam'], jam.paths(parsed), ROOT)


def test_virtualenv_symlink_is_not_resolved(tmp_path):
    binary = tmp_path / 'env/bin/python'
    binary.parent.mkdir(parents=True)
    binary.symlink_to(sys.executable)
    assert python_executable(venv=tmp_path / 'env') == str(binary)
    assert python_executable(str(binary)) == str(binary)


def test_training_failure_is_logged_and_stops_pipeline(tmp_path, monkeypatch):
    parsed = args('train', 'wan-jam', '--output', str(tmp_path / 'failed'))
    plan, env = jam.command_plan(parsed)
    plan['commands'] = [[sys.executable, '-c', 'print("training failed"); raise SystemExit(7)']]
    with pytest.raises(subprocess.CalledProcessError) as caught:
        jam.execute_training(parsed, plan, env)
    assert caught.value.returncode == 7
    assert json.loads((tmp_path / 'failed/run.json').read_text())['status'] == 'failed'
    assert 'training failed' in (tmp_path / 'failed/train_node0.log').read_text()


@pytest.mark.parametrize('outcome', ['success', 'train_failed', 'checkpoint_missing', 'simulator_missing'])
def test_pipeline_execution_order_and_failure_boundaries(tmp_path, monkeypatch, outcome):
    parsed = args('train-eval', 'wan-jam', '--output', str(tmp_path / 'run'))
    plan, env = jam.command_plan(parsed)
    checkpoint = Path(plan['expected_checkpoint'])
    marker = tmp_path / 'evaluation_ran'
    script = (
        'from pathlib import Path; '
        f'p=Path({str(checkpoint)!r}); p.parent.mkdir(parents=True,exist_ok=True); '
        f'Path({str(tmp_path / "run/dataset_stats.json")!r}).write_text("{{}}"); '
    )
    if outcome == 'train_failed':
        script += 'raise SystemExit(9)'
    elif outcome == 'checkpoint_missing':
        script += 'print("stopped before final checkpoint")'
    else:
        script += 'p.write_bytes(b"test double")'
    plan['stages'][0]['commands'] = [[sys.executable, '-c', script]]
    plan['stages'][1]['commands'] = [[sys.executable, '-c',
                                    f'from pathlib import Path; Path({str(marker)!r}).touch()']]
    if outcome != 'simulator_missing':
        (tmp_path / 'third_party/LIBERO/libero/libero/assets').mkdir(parents=True)
        (tmp_path / 'third_party/.libero').mkdir(parents=True)
        (tmp_path / 'third_party/.libero/config.yaml').write_text('benchmark_root: "x"\n')
    monkeypatch.setattr(jam, 'ROOT', tmp_path)
    saved_recipes = json.loads((ROOT / 'reproduction/recipes.json').read_text())['recipes']
    monkeypatch.setattr(jam, 'recipes', lambda: saved_recipes)
    monkeypatch.setattr(jam, 'command_plan', lambda _: (plan, env))
    monkeypatch.setattr(sys, 'argv', ['jam.py', 'train-eval', 'wan-jam', '--output', str(tmp_path / 'run')])
    if outcome == 'success':
        jam.main()
    else:
        with pytest.raises(SystemExit):
            jam.main()
    assert marker.exists() == (outcome == 'success')
    if outcome == 'simulator_missing':
        assert not (tmp_path / 'run').exists()


def evaluation_args(tmp_path, recipe='wan-jam', benchmark='libero'):
    return argparse.Namespace(recipe=recipe, benchmark=benchmark,
        checkpoint=str(tmp_path / 'weights.pt'), stats=str(tmp_path / 'stats.json'),
        gpus='2,5', output=str(tmp_path / 'eval'), data_root=None,
        model_root=None, cache_root=None, override=[])


@pytest.mark.parametrize('recipe', LIBERO_RECIPES)
@pytest.mark.parametrize('benchmark', ['libero', 'plus'])
def test_evaluators_have_identical_task_coverage_with_backend_specific_inference(recipe, benchmark, tmp_path):
    plan = evaluate_libero.build_plan(evaluation_args(tmp_path, recipe, benchmark))
    tasks = [task for item in plan['workers'] for task in item['tasks']]
    assert len(tasks) == len(set(tasks)) == (40 if benchmark == 'libero' else 1506)
    assert set(tasks) == set(task_list(benchmark))
    for worker in plan['workers']:
        command = worker['command']
        assert '++EVALUATION.canonical_result_names=true' in command
        assert '++EVALUATION.num_trials=' + str(trials_per_task(benchmark)) in command
        replan = 10 if recipe.startswith(('wan', 'qwen')) or (recipe.startswith('dino') and benchmark == 'plus') else 8
        assert f'++EVALUATION.replan_steps={replan}' in command
        assert ('++model.skip_dit_load_from_pretrain=false' in command) == recipe.startswith('qwen')
        assert ('++model.tensor_native_preprocessing=false' in command) == recipe.startswith('qwen')
        assert any(x.startswith('++model.qwen3_model_spec=') for x in command) == recipe.startswith('flux')
        assert '++model.action_dit_pretrained_path=null' in command


@pytest.mark.parametrize('benchmark,config', [('libero', '.libero'), ('plus', '.libero_plus'), ('robotwin', None)])
def test_each_libero_benchmark_reads_its_own_configuration(benchmark, config, monkeypatch):
    monkeypatch.delenv('LIBERO_CONFIG_PATH', raising=False)
    env = jam.environment('wan', benchmark, {'models': Path('/models')})
    assert env.get('LIBERO_CONFIG_PATH') == (str(ROOT / 'third_party' / config) if config else None)


@pytest.mark.parametrize('override', ['++EVALUATION.num_trials=1', '++EVALUATION.env_num=8', 'EVALUATION={num_trials:1}',
                                     '++EVALUATION.output_dir=/elsewhere'])
def test_protected_evaluation_contract(override, tmp_path):
    parsed = evaluation_args(tmp_path)
    parsed.override.append(override)
    with pytest.raises(ValueError, match='managed'):
        evaluate_libero.build_plan(parsed)


def write_result(output, key, n=50, successes=25, filename=None):
    suite, tid = key.split(',')
    path = output / suite / (filename or f'task{tid}_results.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'task_suite': suite, 'task_id': int(tid), 'total_episodes': n,
        'successes': successes, 'success_episodes': list(range(successes)),
        'failure_episodes': list(range(successes, n))}))
    return path


def test_complete_libero_and_episode_csv(tmp_path):
    for task in task_list('libero'):
        write_result(tmp_path, task)
    report, episodes = summarize(tmp_path, 'libero')
    assert report['complete'] and report['valid_episodes'] == 2000
    assert report['scores_percent'] == dict.fromkeys(['Spatial','Object','Goal','Long','AVG'], 50)
    save_summary(tmp_path, report, episodes)
    rows = list(csv.DictReader((tmp_path / 'episodes.csv').open()))
    assert len(rows) == 2000 and len({(x['task_id'], x['episode']) for x in rows}) == 2000


def test_plus_uses_task_weighted_average(tmp_path):
    for key in task_list('plus'):
        write_result(tmp_path, key, n=1, successes=int(key.startswith('libero_spatial,')))
    report, _ = summarize(tmp_path, 'plus')
    assert report['complete']
    assert report['success_rate_percent'] == pytest.approx(100 * 361 / 1506)
    assert report['success_rate_percent'] != 25


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'trials', 'episodes', 'malformed', 'unexpected'])
def test_incomplete_or_ambiguous_results_never_get_an_average(tmp_path, corruption):
    for key in task_list('libero'):
        write_result(tmp_path, key)
    first = tmp_path / 'libero_spatial/task0_results.json'
    if corruption == 'missing':
        first.unlink()
    elif corruption == 'duplicate':
        shutil.copyfile(first, first.with_name('gpu0_task0_results.json'))
    elif corruption == 'trials':
        write_result(tmp_path, 'libero_spatial,0', n=1, successes=1)
    elif corruption == 'episodes':
        data = json.loads(first.read_text())
        data['failure_episodes'][-1] = 0
        first.write_text(json.dumps(data))
    elif corruption == 'malformed':
        first.write_text('{')
    else:
        write_result(tmp_path, 'libero_spatial,99')
    report, episodes = summarize(tmp_path, 'libero')
    assert not report['complete'] and report['success_rate_percent'] is None
    assert all(score is None for score in report['scores_percent'].values())
    save_summary(tmp_path, report, episodes)
    assert len(list(csv.DictReader((tmp_path / 'episodes.csv').open()))) == 0


def prepared_evaluation(tmp_path, monkeypatch):
    parsed = evaluation_args(tmp_path)
    plan = evaluate_libero.build_plan(parsed)
    Path(parsed.checkpoint).write_bytes(b'test double')
    Path(parsed.stats).write_text('{}')
    (tmp_path / 'third_party/LIBERO/libero/libero/assets').mkdir(parents=True)
    (tmp_path / 'third_party/.libero').mkdir(parents=True)
    (tmp_path / 'third_party/.libero/config.yaml').write_text('benchmark_root: "x"\n')
    monkeypatch.setattr(evaluate_libero, 'ROOT', tmp_path)
    monkeypatch.setattr(evaluate_libero, 'POLL_SECONDS', 0)
    monkeypatch.setattr(evaluate_libero, 'STALL_SECONDS', 0)
    return parsed, plan


class FakeWorker:
    """A worker process that has exited with `code`, or hangs until it is killed."""

    def __init__(self, code, hang):
        self.pid, self.returncode = -1, None if hang else code

    def poll(self):
        return self.returncode

    def wait(self):
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


def fake_workers(monkeypatch, behaviour):
    """behaviour(call, tasks) -> (tasks it finishes, exit code, hangs); returns the task list of every call."""
    calls, killed = [], []
    calls_lock = Lock()

    def popen(command, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] in ('2', '5') and kwargs['stdin'] is subprocess.DEVNULL
        value = lambda key: json.loads(next(x for x in command if x.startswith(key + '=')).split('=', 1)[1])
        tasks = Path(value('++EVALUATION.task_file')).read_text().splitlines()
        with calls_lock:
            calls.append(tasks)
            call = len(calls)
        finish, code, hang = behaviour(call, tasks)
        for task in finish:
            write_result(Path(value('++EVALUATION.output_dir')), task)
        return FakeWorker(code, hang)

    monkeypatch.setattr(evaluate_libero.subprocess, 'Popen', popen)
    monkeypatch.setattr(evaluate_libero.os, 'killpg', lambda pid, sig: killed.append(pid))
    return calls, killed


@pytest.mark.parametrize('worker_failure', [False, True])
def test_shared_evaluation_execution_with_fake_workers(tmp_path, monkeypatch, worker_failure):
    parsed, plan = prepared_evaluation(tmp_path, monkeypatch)
    calls, _ = fake_workers(monkeypatch, lambda call, tasks: ([] if worker_failure else tasks, int(worker_failure), False))
    assert evaluate_libero.execute(parsed, plan) == int(worker_failure)
    report = json.loads((Path(parsed.output) / 'execution_report.json').read_text())
    assert report['results_complete'] is not worker_failure
    assert report['attempts'] == (1 + evaluate_libero.RETRIES if worker_failure else 1)
    assert report['success_rate_percent'] == (None if worker_failure else 50)
    summary, _ = summarize(Path(parsed.output), 'libero')
    assert summary['complete'] is not worker_failure
    with pytest.raises(ValueError, match='not empty'):
        evaluate_libero.execute(parsed, plan)


def test_crashed_and_stalled_workers_rerun_only_their_unfinished_tasks(tmp_path, monkeypatch):
    parsed, plan = prepared_evaluation(tmp_path, monkeypatch)

    def behaviour(call, tasks):
        if call == 1:  # crashes before its last task
            return tasks[:-1], 1, False
        if call == 2:  # hangs without finishing anything
            return [], None, True
        return tasks, 0, False

    calls, killed = fake_workers(monkeypatch, behaviour)
    assert evaluate_libero.execute(parsed, plan) == 0
    first = [worker['tasks'] for worker in plan['workers']]
    assert sorted(calls[:2]) == sorted(first) and len(killed) == len(calls)
    assert sorted(sum(calls[2:], [])) == sorted([calls[0][-1]] + calls[1])
    report = json.loads((Path(parsed.output) / 'execution_report.json').read_text())
    assert report['results_complete'] and report['attempts'] == 2
    assert (Path(parsed.output) / 'worker_0_retry1.log').is_file()


def test_watchdog_retries_when_worker_exits_before_kill(tmp_path, monkeypatch):
    parsed, plan = prepared_evaluation(tmp_path, monkeypatch)
    calls, _ = fake_workers(monkeypatch, lambda call, tasks: ([], None, True))

    def already_exited(pid, sig):
        raise ProcessLookupError('worker exited after poll')

    monkeypatch.setattr(evaluate_libero.os, 'killpg', already_exited)
    assert evaluate_libero.execute(parsed, plan) == 1
    report = json.loads((Path(parsed.output) / 'execution_report.json').read_text())
    assert report['attempts'] == 1 + evaluate_libero.RETRIES
    assert len(calls) == len(plan['workers']) * (1 + evaluate_libero.RETRIES)


def test_resume_keeps_valid_results_and_refuses_another_evaluation(tmp_path, monkeypatch):
    parsed, plan = prepared_evaluation(tmp_path, monkeypatch)
    output = Path(parsed.output)
    output.mkdir()
    (output / 'evaluation_plan.json').write_text(json.dumps(plan))
    done = task_list('libero')[:30]
    for task in done:
        write_result(output, task)
    broken = write_result(output, task_list('libero')[30])
    broken.write_text('{')
    calls, _ = fake_workers(monkeypatch, lambda call, tasks: (tasks, 0, False))
    with pytest.raises(ValueError, match='not empty'):
        evaluate_libero.execute(parsed, plan)
    parsed.resume = True
    assert evaluate_libero.execute(parsed, plan) == 0
    assert sorted(sum(calls, [])) == sorted(task_list('libero')[30:])
    assert broken.with_name(broken.name + '.invalid').is_file()
    other = copy.deepcopy(plan)
    other['checkpoint'] = str(tmp_path / 'other.pt')
    Path(other['checkpoint']).write_bytes(b'another checkpoint')
    with pytest.raises(ValueError, match='different evaluation'):
        evaluate_libero.execute(parsed, other)


def test_conflicting_duplicate_results_are_both_rerun(tmp_path, monkeypatch):
    parsed, plan = prepared_evaluation(tmp_path, monkeypatch)
    parsed.resume = True
    output = Path(parsed.output)
    output.mkdir()
    (output / 'evaluation_plan.json').write_text(json.dumps(plan))
    for task in task_list('libero'):
        write_result(output, task)
    key = 'libero_spatial,0'
    conflicting = write_result(output, key, successes=0, filename='other_task0_results.json')
    canonical = output / 'libero_spatial/task0_results.json'
    calls, _ = fake_workers(monkeypatch, lambda call, tasks: (tasks, 0, False))
    assert evaluate_libero.execute(parsed, plan) == 0
    assert sum(calls, []) == [key]
    assert canonical.with_name(canonical.name + '.invalid').is_file()
    assert conflicting.with_name(conflicting.name + '.invalid').is_file()
    assert json.loads((output / 'summary.json').read_text())['scores_percent']['Spatial'] == 50


def test_repeated_invalid_results_preserve_prior_evidence(tmp_path):
    path = write_result(tmp_path, 'libero_spatial,0')
    path.write_text('{')
    evaluate_libero.unfinished_tasks(tmp_path, 'libero')
    path.write_text('second broken attempt')
    evaluate_libero.unfinished_tasks(tmp_path, 'libero')
    assert path.with_name(path.name + '.invalid').read_text() == '{'
    assert path.with_name(path.name + '.invalid.1').read_text() == 'second broken attempt'


def test_resume_rejects_changed_resolved_model_path_but_allows_gpu_sharding(tmp_path, monkeypatch):
    parsed, _ = prepared_evaluation(tmp_path, monkeypatch)
    parsed.model_root = str(tmp_path / 'model_a')
    plan = evaluate_libero.build_plan(parsed)
    output = Path(parsed.output)
    output.mkdir()
    (output / 'evaluation_plan.json').write_text(json.dumps(plan))
    for task in task_list('libero'):
        write_result(output, task)
    parsed.resume = True
    parsed.model_root = str(tmp_path / 'model_b')
    changed = evaluate_libero.build_plan(parsed)
    assert all(plan[key] == changed[key] for key in evaluate_libero.RESUME_KEYS)
    with pytest.raises(ValueError, match='different evaluation'):
        evaluate_libero.execute(parsed, changed)
    parsed.model_root = str(tmp_path / 'model_a')
    parsed.gpus = '5,2'
    assert evaluate_libero.execute(parsed, evaluate_libero.build_plan(parsed)) == 0


def _process_running(pid):
    # A killed orphan can briefly remain as a zombie until init reaps it.
    status = Path(f'/proc/{pid}/stat')
    try:
        if status.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
            return False
    except FileNotFoundError:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process-group lifecycle check')
def test_crashed_worker_does_not_leave_its_simulator_process(tmp_path, monkeypatch):
    parsed, plan = prepared_evaluation(tmp_path, monkeypatch)
    (tmp_path / 'backends/wan').mkdir(parents=True)
    monkeypatch.setattr(evaluate_libero, 'RETRIES', 0)
    monkeypatch.setattr(evaluate_libero, 'POLL_SECONDS', 0.01)
    monkeypatch.setattr(evaluate_libero, 'STALL_SECONDS', 30)
    worker = tmp_path / 'worker.py'
    pids = tmp_path / 'pids.json'
    worker.write_text(
        'import json,os,subprocess,sys\nfrom pathlib import Path\n'
        'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
        f'Path({str(pids)!r}).write_text(json.dumps(dict(worker=os.getpid(),child=child.pid)))\n'
        'os._exit(7)\n')
    plan['command'] = [sys.executable, str(worker)]
    try:
        assert evaluate_libero.execute(parsed, plan) == 1
        ids = json.loads(pids.read_text())
        assert not _process_running(ids['child'])
        report = json.loads((Path(parsed.output) / 'execution_report.json').read_text())
        assert report['workers'][0]['exit_code'] == 7
    finally:
        if pids.exists():
            try:
                os.killpg(json.loads(pids.read_text())['worker'], signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process-group lifecycle check')
def test_interrupt_cancels_workers_before_waiting_for_threads(tmp_path):
    (tmp_path / 'third_party/LIBERO/libero/libero/assets').mkdir(parents=True)
    (tmp_path / 'third_party/.libero').mkdir(parents=True)
    (tmp_path / 'third_party/.libero/config.yaml').write_text('benchmark_root: x\n')
    (tmp_path / 'backends/wan').mkdir(parents=True)
    (tmp_path / 'weights.pt').write_bytes(b'test double')
    (tmp_path / 'stats.json').write_text('{}')
    pids = tmp_path / 'pids.json'
    worker = tmp_path / 'worker.py'
    worker.write_text(
        'import json,os,time\nfrom pathlib import Path\n'
        f'Path({str(pids)!r}).write_text(json.dumps(dict(worker=os.getpid())))\n'
        'time.sleep(60)\n')
    runner = tmp_path / 'runner.py'
    runner.write_text(
        f'import argparse,sys\nsys.path.insert(0,{str(ROOT / "tools")!r})\n'
        'from pathlib import Path\nimport evaluate_libero as e\n'
        f'root=Path({str(tmp_path)!r})\n'
        'a=argparse.Namespace(recipe="wan-jam",benchmark="libero",checkpoint=str(root/"weights.pt"),'
        'stats=str(root/"stats.json"),gpus="0",output=str(root/"eval"),'
        'data_root=None,model_root=None,cache_root=None,override=[],resume=False)\n'
        'plan=e.build_plan(a)\n'
        f'plan["command"]=[sys.executable,{str(worker)!r}]\n'
        'e.ROOT=root\ne.POLL_SECONDS=60\ne.execute(a,plan)\n')
    with (tmp_path / 'runner.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(runner)], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while not pids.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert pids.exists(), (tmp_path / 'runner.log').read_text()
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=5) != 0
            assert not _process_running(json.loads(pids.read_text())['worker'])
        finally:
            if pids.exists():
                try:
                    os.killpg(json.loads(pids.read_text())['worker'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.kill()
            process.wait()


def test_libero_plus_assets_are_unpacked_from_their_nested_folder(tmp_path):
    import zipfile
    import setup_simulators
    archive = tmp_path / 'assets.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('upstream/dataset/LIBERO-plus-0/assets/scenes/table.xml', '<mujoco/>')
        z.writestr('upstream/dataset/LIBERO-plus-0/README.md', 'ignored')
    setup_simulators.extract_assets(archive, tmp_path / 'assets', dry_run=False)
    assert [p.relative_to(tmp_path) for p in tmp_path.rglob('*') if p.is_file() and p != archive] == [Path('assets/scenes/table.xml')]


@pytest.mark.parametrize('backend', ['wan', 'multimodal'])
def test_robotwin_portable_task_discovery(backend):
    path = ROOT / 'backends' / backend / 'experiments/robotwin/run_robotwin_manager.py'
    spec = importlib.util.spec_from_file_location('_test_robotwin_' + backend, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tasks = module._load_all_tasks(ROOT / 'third_party/RoboTwin')
    assert len(tasks) == len(set(tasks)) == 50
    plan, _ = jam.command_plan(args('eval', f'robotwin-{ "wan" if backend == "wan" else "ovis" }-jam',
                                    '--checkpoint', 'ckpt.pt', '--stats', 'stats.json'))
    assert '++EVALUATION.exact_output_dir=true' in plan['commands'][0]
    assert plan['output'].endswith('robotwin/ckpt')


@pytest.mark.parametrize('recipe,benchmark', [(recipe, benchmark) for recipe in LIBERO_RECIPES
                                               for benchmark in ('libero', 'plus')]
                         + [(recipe, 'robotwin') for recipe in ROBOTWIN_RECIPES])
def test_evaluation_commands_compose_without_loading_backends(recipe, benchmark, tmp_path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    r = jam.recipes()[recipe]
    repo = ROOT / 'backends' / r['backend']
    spec = importlib.util.spec_from_file_location('_workflow_config_resolvers',
                                                repo / 'src' / jam.backend_package(r['backend']) / 'utils/config_resolvers.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register_default_resolvers()
    if benchmark == 'robotwin':
        plan, _ = jam.command_plan(args('eval', recipe, '--benchmark', benchmark,
                                        '--checkpoint', 'weights.pt', '--stats', 'stats.json'))
        command = plan['commands'][0]
    else:
        plan = evaluate_libero.build_plan(evaluation_args(tmp_path, recipe, benchmark))
        command = plan['workers'][0]['command']
    index = command.index('--config-name')
    with initialize_config_dir(config_dir=str(repo / 'configs'), version_base='1.3'):
        config = compose(config_name=command[index + 1], overrides=command[index + 2:])
        resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert resolved['ckpt'].endswith('weights.pt')
    assert resolved['EVALUATION']['dataset_stats_path'].endswith('stats.json')
