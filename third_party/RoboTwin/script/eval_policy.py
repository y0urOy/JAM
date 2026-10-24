import sys
import os
import subprocess
import json
import hashlib
import math
import time

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


_EVAL_PROGRESS_VERSION = 1
_EVAL_LIVENESS_VERSION = 1
_MAX_ATTEMPTS_PER_SEED = 3
_eval_liveness_sequence = 0
_eval_liveness_last_write = 0.0


def _is_renderer_device_lost(error):
    """Return true for fatal Vulkan device-loss errors.

    Continuing to call ``close_env`` after Vulkan has lost the device can hang
    inside native cleanup.  Let the torch-first wrapper hard-exit instead; a
    restarted run resumes from the saved in-flight seed.
    """
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = f"{type(current).__name__}: {current}".lower()
        compact = message.replace("_", "").replace(" ", "")
        if "errordevicelost" in compact or "vkerrordevicelost" in compact:
            return True
        current = current.__cause__ or current.__context__
    return False


def _atomic_write_json(path, payload):
    """Atomically persist resumable evaluation progress."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
    os.replace(tmp_path, path)


def _load_paired_eval_phase(
    manifest_path,
    *,
    task_name,
    phase,
    test_num,
    instruction_type,
    initial_seed,
):
    """Load prevalidated seed+instruction pairs for a fast paired ablation."""
    path = Path(str(manifest_path)).expanduser().resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)
    if int(payload.get("version", -1)) != 1:
        raise RuntimeError(f"Unsupported paired manifest version: {path}")
    protocol = payload.get("protocol", {})
    manifest_episodes = int(protocol.get("episodes_per_phase", -1))
    if manifest_episodes < int(test_num):
        raise RuntimeError(
            f"Paired manifest has too few episodes: manifest={manifest_episodes} "
            f"requested={test_num}"
        )
    if str(protocol.get("instruction_type")) != str(instruction_type):
        raise RuntimeError(
            "Paired manifest instruction type mismatch: "
            f"manifest={protocol.get('instruction_type')!r} requested={instruction_type!r}"
        )
    try:
        entries = payload["tasks"][str(task_name)][str(phase)]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"Paired manifest has no task/phase {task_name}/{phase}: {path}"
        ) from error
    if not isinstance(entries, list) or len(entries) < int(test_num):
        raise RuntimeError(
            f"Paired manifest {task_name}/{phase} must contain at least {test_num} entries"
        )
    normalized = []
    seeds = set()
    for expected_idx, entry in enumerate(entries[: int(test_num)]):
        episode_idx = int(entry.get("episode_idx", -1))
        seed = int(entry.get("seed", -1))
        instruction = str(entry.get("instruction", "")).strip()
        if episode_idx != expected_idx or seed < int(initial_seed) or not instruction:
            raise RuntimeError(
                f"Invalid paired entry {task_name}/{phase}/{expected_idx}: {entry!r}"
            )
        if seed in seeds:
            raise RuntimeError(
                f"Duplicate paired seed {seed} in {task_name}/{phase}: {path}"
            )
        seeds.add(seed)
        normalized.append(
            {"episode_idx": episode_idx, "seed": seed, "instruction": instruction}
        )
    print(
        "ROBOTWIN_PAIRED_MANIFEST "
        f"path={path} sha256={digest} task={task_name} phase={phase} "
        f"episodes={len(normalized)} expert_rollout=false",
        flush=True,
    )
    return normalized, digest


def _write_eval_liveness(
    stage,
    *,
    episode=None,
    seed=None,
    action_count=None,
    force=False,
):
    """Publish opt-in process liveness without changing evaluation progress.

    The caller chooses where this file lives.  It is deliberately
    separate from the scientific progress file: liveness can reset a native
    hang watchdog, but it can never advance an episode, seed, or success count.
    """
    path = os.environ.get("ROBOTWIN_EVAL_LIVENESS_PATH")
    if not path:
        return

    raw_interval = os.environ.get("ROBOTWIN_EVAL_LIVENESS_INTERVAL_SECONDS", "5")
    try:
        interval = float(raw_interval)
    except ValueError as error:
        raise ValueError(
            "ROBOTWIN_EVAL_LIVENESS_INTERVAL_SECONDS must be numeric, "
            f"got {raw_interval!r}"
        ) from error
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError(
            "ROBOTWIN_EVAL_LIVENESS_INTERVAL_SECONDS must be finite and positive, "
            f"got {interval}"
        )

    global _eval_liveness_sequence, _eval_liveness_last_write
    now = time.monotonic()
    if not force and now - _eval_liveness_last_write < interval:
        return
    _eval_liveness_sequence += 1
    _eval_liveness_last_write = now
    _atomic_write_json(
        path,
        {
            "version": _EVAL_LIVENESS_VERSION,
            "pid": os.getpid(),
            "sequence": _eval_liveness_sequence,
            "stage": str(stage),
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "episode": None if episode is None else int(episode),
            "seed": None if seed is None else int(seed),
            "action_count": None if action_count is None else int(action_count),
        },
    )


def _load_eval_progress(path, expected_metadata, initial_seed, test_num):
    path = Path(path)
    if not path.is_file():
        return {
            "next_seed": int(initial_seed),
            "completed_episodes": 0,
            "successes": 0,
            "inflight_seed": None,
            "seed_attempt": 0,
        }

    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("version") != _EVAL_PROGRESS_VERSION:
        raise RuntimeError(f"Unsupported RoboTwin progress version in {path}: {payload}")
    if payload.get("metadata") != expected_metadata:
        raise RuntimeError(
            f"RoboTwin progress metadata mismatch in {path}: "
            f"saved={payload.get('metadata')!r}, expected={expected_metadata!r}"
        )

    state = payload.get("state", {})
    completed = int(state.get("completed_episodes", -1))
    successes = int(state.get("successes", -1))
    next_seed = int(state.get("next_seed", -1))
    inflight_seed = state.get("inflight_seed")
    if inflight_seed is not None:
        inflight_seed = int(inflight_seed)
    seed_attempt = int(state.get("seed_attempt", 0))
    if not 0 <= successes <= completed <= int(test_num):
        raise RuntimeError(f"Invalid RoboTwin progress counters in {path}: {state}")
    if next_seed < int(initial_seed) or seed_attempt < 0:
        raise RuntimeError(f"Invalid RoboTwin progress seed state in {path}: {state}")
    if inflight_seed is not None and inflight_seed != next_seed:
        raise RuntimeError(f"Inconsistent RoboTwin inflight seed in {path}: {state}")
    return {
        "next_seed": next_seed,
        "completed_episodes": completed,
        "successes": successes,
        "inflight_seed": inflight_seed,
        "seed_attempt": seed_attempt,
    }


def _save_eval_progress(
    path,
    metadata,
    *,
    next_seed,
    completed_episodes,
    successes,
    inflight_seed=None,
    seed_attempt=0,
):
    _atomic_write_json(
        path,
        {
            "version": _EVAL_PROGRESS_VERSION,
            "metadata": metadata,
            "state": {
                "next_seed": int(next_seed),
                "completed_episodes": int(completed_episodes),
                "successes": int(successes),
                "inflight_seed": None if inflight_seed is None else int(inflight_seed),
                "seed_attempt": int(seed_attempt),
            },
        },
    )


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_eval_video_size(args):
    head_camera_cfg = get_camera_config(args["camera"]["head_camera_type"])
    video_w = int(head_camera_cfg["w"])
    video_h = int(head_camera_cfg["h"])

    if args["camera"].get("collect_wrist_camera", False):
        wrist_camera_cfg = get_camera_config(args["camera"]["wrist_camera_type"])
        wrist_w = int(wrist_camera_cfg["w"])
        wrist_h = int(wrist_camera_cfg["h"])
        video_w = max(video_w, wrist_w * 2)
        video_h = video_h + wrist_h

    return f"{video_w}x{video_h}"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported `task_config` for fixed result naming: {task_config}. "
        "Expected one of: ['demo_clean', 'demo_randomized']."
    )


def main(usr_args):
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    eval_output_dir = usr_args.get("eval_output_dir")
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Evaluation videos are diagnostic artifacts.  Metric-only launchers can
    # skip one ffmpeg encoder per episode without changing simulator seeds,
    # observations, policy actions, environment transitions, or scoring.
    if "eval_video_log" in usr_args:
        args["eval_video_log"] = parse_bool(usr_args["eval_video_log"])
    print(
        f"ROBOTWIN_EVAL_VIDEO_LOG enabled={bool(args.get('eval_video_log', False))}",
        flush=True,
    )

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{eval_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)
    # Clean and randomized phases for one task can run concurrently.  Isolate mutable ffmpeg artifacts so both
    # writers can never race on ``episode<N>.mp4``.
    result_suffix = _result_suffix_from_task_config(task_config)

    if args["eval_video_log"]:
        video_save_dir = save_dir / f"videos_{result_suffix}"
        video_size = get_eval_video_size(args)
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = eval_num_episodes
    topk = 1

    paired_episodes = None
    paired_manifest_sha256 = None
    paired_manifest_path = usr_args.get("paired_manifest_path")
    if paired_manifest_path not in (None, "", "None", "null"):
        paired_episodes, paired_manifest_sha256 = _load_paired_eval_phase(
            paired_manifest_path,
            task_name=task_name,
            phase=result_suffix,
            test_num=test_num,
            instruction_type=instruction_type,
            initial_seed=st_seed,
        )

    file_path = Path(save_dir) / f"_result_{result_suffix}.txt"
    progress_path = Path(save_dir) / f"_progress_{result_suffix}.json"
    progress_identity = os.environ.get(
        "ROBOTWIN_EVAL_PROGRESS_ID", str(ckpt_setting)
    )
    progress_metadata = {
        "task_name": str(task_name),
        "task_config": str(task_config),
        "ckpt_setting": str(progress_identity),
        "instruction_type": str(instruction_type),
        "eval_num_episodes": int(test_num),
        "initial_seed": int(st_seed),
        "paired_manifest_sha256": paired_manifest_sha256,
    }

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   skip_get_obs_within_replan=skip_get_obs_within_replan,
                                   progress_path=progress_path,
                                   progress_metadata=progress_metadata,
                                   paired_episodes=paired_episodes)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    result_payload = (
        f"Timestamp: {eval_ts}\n\n"
        f"Instruction Type: {instruction_type}\n\n"
        + "\n".join(map(str, np.array(suc_nums) / test_num))
    )
    result_tmp = file_path.with_name(f".{file_path.name}.tmp.{os.getpid()}")
    with open(result_tmp, "w", encoding="utf-8") as file:
        file.write(result_payload)
        file.flush()
    os.replace(result_tmp, file_path)
    progress_path.unlink(missing_ok=True)

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                skip_get_obs_within_replan=False,
                progress_path=None,
                progress_metadata=None,
                paired_episodes=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    paired_mode = paired_episodes is not None
    expert_check = not paired_mode
    if progress_path is None or progress_metadata is None:
        raise ValueError("Resumable RoboTwin evaluation requires progress path and metadata")
    progress_initial_seed = (
        int(paired_episodes[0]["seed"]) if paired_mode else int(st_seed)
    )
    progress = _load_eval_progress(
        progress_path, progress_metadata, progress_initial_seed, test_num
    )
    now_id = progress["completed_episodes"]
    succ_seed = now_id
    TASK_ENV.suc = progress["successes"]
    TASK_ENV.test_num = now_id

    now_seed = progress["next_seed"]
    if paired_mode and now_id < test_num:
        expected_seed = int(paired_episodes[now_id]["seed"])
        if int(now_seed) != expected_seed:
            raise RuntimeError(
                "Paired progress/manifest seed mismatch: "
                f"completed={now_id} progress_seed={now_seed} expected={expected_seed}"
            )
    seed_attempt = progress["seed_attempt"]
    if progress["inflight_seed"] is not None:
        print(
            f"RESUME_INFLIGHT seed={now_seed} attempt={seed_attempt} "
            f"completed={now_id}/{test_num}",
            flush=True,
        )
        if seed_attempt >= _MAX_ATTEMPTS_PER_SEED and paired_mode:
            print(
                f"RETRY_PAIRED_RENDERER_UNSTABLE_SEED seed={now_seed} "
                f"attempts={seed_attempt}",
                flush=True,
            )
            seed_attempt = 0
            _save_eval_progress(
                progress_path,
                progress_metadata,
                next_seed=now_seed,
                completed_episodes=now_id,
                successes=TASK_ENV.suc,
            )
        elif seed_attempt >= _MAX_ATTEMPTS_PER_SEED:
            print(
                f"SKIP_RENDERER_UNSTABLE_SEED seed={now_seed} attempts={seed_attempt}",
                flush=True,
            )
            now_seed += 1
            seed_attempt = 0
            _save_eval_progress(
                progress_path,
                progress_metadata,
                next_seed=now_seed,
                completed_episodes=now_id,
                successes=TASK_ENV.suc,
            )

    print(
        f"ROBOTWIN_RESUME completed={now_id}/{test_num} successes={TASK_ENV.suc} "
        f"next_seed={now_seed}",
        flush=True,
    )
    _write_eval_liveness(
        "phase_resume", episode=now_id, seed=now_seed, force=True
    )
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        paired_entry = paired_episodes[now_id] if paired_mode else None
        if paired_entry is not None:
            expected_seed = int(paired_entry["seed"])
            if int(now_seed) != expected_seed:
                raise RuntimeError(
                    f"Paired seed drift at episode {now_id}: "
                    f"current={now_seed} expected={expected_seed}"
                )
        seed_attempt += 1
        _save_eval_progress(
            progress_path,
            progress_metadata,
            next_seed=now_seed,
            completed_episodes=now_id,
            successes=TASK_ENV.suc,
            inflight_seed=now_seed,
            seed_attempt=seed_attempt,
        )
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                _write_eval_liveness(
                    "expert_setup_start", episode=now_id, seed=now_seed, force=True
                )
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                _write_eval_liveness(
                    "expert_setup_done", episode=now_id, seed=now_seed, force=True
                )
                _write_eval_liveness(
                    "expert_play_start", episode=now_id, seed=now_seed, force=True
                )
                episode_info = TASK_ENV.play_once()
                _write_eval_liveness(
                    "expert_play_done", episode=now_id, seed=now_seed, force=True
                )
                _write_eval_liveness(
                    "expert_close_start", episode=now_id, seed=now_seed, force=True
                )
                TASK_ENV.close_env()
                _write_eval_liveness(
                    "expert_close_done", episode=now_id, seed=now_seed, force=True
                )
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                seed_attempt = 0
                _save_eval_progress(
                    progress_path,
                    progress_metadata,
                    next_seed=now_seed,
                    completed_episodes=now_id,
                    successes=TASK_ENV.suc,
                )
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                if _is_renderer_device_lost(e):
                    print(
                        f"FATAL_RENDERER_DEVICE_LOST seed={now_seed}; "
                        "restarting from saved progress",
                        flush=True,
                    )
                    raise
                print(" -------------")
                print("Error: ", e)
                print("Stack Trace: ", traceback.format_exc())
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                seed_attempt = 0
                _save_eval_progress(
                    progress_path,
                    progress_metadata,
                    next_seed=now_seed,
                    completed_episodes=now_id,
                    successes=TASK_ENV.suc,
                )
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            seed_attempt = 0
            _save_eval_progress(
                progress_path,
                progress_metadata,
                next_seed=now_seed,
                completed_episodes=now_id,
                successes=TASK_ENV.suc,
            )
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        try:
            _write_eval_liveness(
                "rollout_setup_start", episode=now_id, seed=now_seed, force=True
            )
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            _write_eval_liveness(
                "rollout_setup_done", episode=now_id, seed=now_seed, force=True
            )
        except UnStableError as e:
            if paired_mode:
                TASK_ENV.close_env()
                raise RuntimeError(
                    f"Paired rollout setup was unstable for required seed {now_seed}"
                ) from e
            # This seed passed expert_check but failed during rollout env init.
            # Roll back the accepted-seed counter and skip to next seed.
            succ_seed -= 1
            if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            TASK_ENV.close_env()
            now_seed += 1
            seed_attempt = 0
            _save_eval_progress(
                progress_path,
                progress_metadata,
                next_seed=now_seed,
                completed_episodes=now_id,
                successes=TASK_ENV.suc,
            )
            continue
        except Exception as e:
            if _is_renderer_device_lost(e):
                print(
                    f"FATAL_RENDERER_DEVICE_LOST seed={now_seed}; "
                    "restarting from saved progress",
                    flush=True,
                )
                raise
            if paired_mode:
                print(" -------------")
                print("Paired rollout setup error: ", e)
                print("Stack Trace: ", traceback.format_exc())
                print(" -------------")
                TASK_ENV.close_env()
                raise RuntimeError(
                    f"Paired rollout setup failed for required seed {now_seed}"
                ) from e
            succ_seed -= 1
            if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            print(" -------------")
            print("Error: ", e)
            print("Stack Trace: ", traceback.format_exc())
            print(" -------------")
            TASK_ENV.close_env()
            now_seed += 1
            seed_attempt = 0
            _save_eval_progress(
                progress_path,
                progress_metadata,
                next_seed=now_seed,
                completed_episodes=now_id,
                successes=TASK_ENV.suc,
            )
            print("error occurs !")
            continue
        if paired_entry is not None:
            instruction = str(paired_entry["instruction"])
        else:
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(
                args["task_name"], episode_info_list, test_num
            )
            instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        current_video_path = None
        if TASK_ENV.eval_video_path is not None:
            episode_idx = TASK_ENV.test_num
            current_video_path = Path(TASK_ENV.eval_video_path) / f"episode{episode_idx}.mp4"
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    str(current_video_path),
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        _write_eval_liveness(
            "model_reset_start", episode=now_id, seed=now_seed, force=True
        )
        reset_func(model)
        _write_eval_liveness(
            "model_reset_done", episode=now_id, seed=now_seed, force=True
        )
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            action_count = TASK_ENV.take_action_cnt
            _write_eval_liveness(
                "rollout_step_start",
                episode=now_id,
                seed=now_seed,
                action_count=action_count,
            )
            need_obs = True
            if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                need_obs = bool(model.should_request_observation())

            observation = None
            if need_obs:
                observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            _write_eval_liveness(
                "rollout_step_done",
                episode=now_id,
                seed=now_seed,
                action_count=TASK_ENV.take_action_cnt,
            )
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            _write_eval_liveness(
                "video_close_start", episode=now_id, seed=now_seed, force=True
            )
            TASK_ENV._del_eval_video_ffmpeg()
            _write_eval_liveness(
                "video_close_done", episode=now_id, seed=now_seed, force=True
            )
            if current_video_path is None or not current_video_path.exists():
                raise FileNotFoundError(f"Expected eval video file not found: {current_video_path}")
            is_randomized = "randomized" in str(args["task_config"]).lower()
            renamed_video_path = (
                Path(TASK_ENV.eval_video_path)
                / f"episode{episode_idx}_randomized-{str(is_randomized).lower()}_success-{str(succ).lower()}.mp4"
            )
            current_video_path.rename(renamed_video_path)

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        completed_seed = int(now_seed)
        now_id += 1
        _write_eval_liveness(
            "rollout_close_start", episode=now_id - 1, seed=now_seed, force=True
        )
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        _write_eval_liveness(
            "rollout_close_done", episode=now_id - 1, seed=now_seed, force=True
        )

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        if paired_mode and now_id < test_num:
            now_seed = int(paired_episodes[now_id]["seed"])
        else:
            now_seed = completed_seed + 1
        seed_attempt = 0
        _save_eval_progress(
            progress_path,
            progress_metadata,
            next_seed=now_seed,
            completed_episodes=now_id,
            successes=TASK_ENV.suc,
        )
        _write_eval_liveness(
            "episode_complete", episode=now_id, seed=completed_seed, force=True
        )

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
