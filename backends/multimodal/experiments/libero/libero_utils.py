"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import multiprocessing as mp
import time
import pathlib

import imageio
from PIL import Image, ImageDraw
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from jam_multimodal.utils.video_io import save_mp4

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


class _SparseRenderOffScreenEnv(OffScreenRenderEnv):
    """Skip camera sensors on steps whose pixels are never consumed.

    LIBERO returns two rendered camera observations after every action.  The
    policy, however, only consumes pixels immediately before a replan.  A
    mapping payload lets the evaluator mark intermediate steps so MuJoCo still
    advances identically while the passive camera render calls are replaced
    with cached reads.
    """

    _PAYLOAD_ACTION_KEY = "imagewam_action"
    _PAYLOAD_RENDER_KEY = "render_observation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._camera_sensor_modes = {}
        for name, observable in self.env._observables.items():
            if observable.modality != "image":
                continue
            original_sensor = observable._sensor

            def cached_sensor(obs_cache, *, _observable=observable):
                del obs_cache
                return _observable._current_observed_value

            # Observable.modality is delegated to this sensor attribute.
            cached_sensor.__modality__ = original_sensor.__modality__
            self._camera_sensor_modes[name] = (original_sensor, cached_sensor)

    def _set_camera_observables_enabled(self, enabled):
        for name, (original_sensor, cached_sensor) in self._camera_sensor_modes.items():
            observable = self.env._observables[name]
            # Keep the observable enabled so its sampling clock advances
            # exactly as in the reference path. On skipped steps its cheap
            # sensor reuses the last frame and is inactive, avoiding both
            # rendering and image IPC. Restoring the real sensor on a replan
            # step therefore preserves the original sampling phase exactly.
            observable._sensor = original_sensor if enabled else cached_sensor
            observable.set_active(enabled)

    def step(self, action):
        render_observation = True
        if isinstance(action, dict) and self._PAYLOAD_ACTION_KEY in action:
            render_observation = bool(action.get(self._PAYLOAD_RENDER_KEY, True))
            action = action[self._PAYLOAD_ACTION_KEY]
        self._set_camera_observables_enabled(render_observation)
        return super().step(action)

    def set_init_state(self, init_state):
        # Always expose a complete initial observation.  The evaluator may
        # subsequently suppress warm-up frames that it does not consume.
        self._set_camera_observables_enabled(True)
        return super().set_init_state(init_state)


class _SingleSubprocEnv:
    """Expose a scalar env API while MuJoCo lives in a spawned subprocess.

    Keeping EGL in a different process from the CUDA policy avoids driver
    context interference without changing observations, actions, or episode
    semantics.  ``SubprocVectorEnv`` already implements the IPC protocol; this
    adapter only unwraps its length-one batch dimension.
    """

    def __init__(self, env):
        self._env = env

    @staticmethod
    def _first(value):
        if isinstance(value, tuple):
            return tuple(_SingleSubprocEnv._first(item) for item in value)
        if isinstance(value, list):
            return value[0]
        if isinstance(value, np.ndarray):
            return value[0]
        return value

    def reset(self):
        return self._first(self._env.reset())

    def set_init_state(self, init_state):
        return self._first(self._env.set_init_state([init_state]))

    def step(self, action):
        return self._first(self._env.step(np.asarray([action])))

    def seed(self, seed):
        result = self._env.seed([seed])
        return result[0]

    def close(self):
        return self._env.close()


def get_libero_env(
    task,
    resolution,
    seed,
    env_num=1,
    force_subprocess=False,
    sparse_camera_observations=False,
):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env_type = _SparseRenderOffScreenEnv if sparse_camera_observations else OffScreenRenderEnv
    if force_subprocess:
        if env_num != 1:
            raise ValueError(
                "force_subprocess currently provides the scalar env API and "
                f"requires env_num=1, got {env_num}."
            )
        # Spawn is required because the parent has already initialized CUDA.
        # Forking that process would inherit an unsafe CUDA runtime state.
        mp.set_start_method("spawn", force=True)
        env = _SingleSubprocEnv(
            SubprocVectorEnv([lambda: env_type(**env_args)])
        )
    elif env_num > 1:
        # The policy process has already initialized CUDA. Use spawn so the
        # software-rendering workers do not inherit that runtime state.
        mp.set_start_method("spawn", force=True)
        env = SubprocVectorEnv([lambda: env_type(**env_args) for _ in range(env_num)])
    else:
        env = env_type(**env_args)
    if env_num > 1 and not force_subprocess:
        # Match scalar evaluation: every trial uses the same simulator seed,
        # while the benchmark-provided initial state selects the episode.
        env.seed([seed] * env_num)
    else:
        env.seed(
            seed
        )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_image(obs):
    """Extracts image from observations and preprocesses it."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    
    # [yc] wrist image
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    
    return {
        "image": img,
        "wrist_image": wrist_img
    }

def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description, log_file=None, fps=24):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        if isinstance(img, dict):
            image = []
            for key, value in img.items():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                pil_img = Image.fromarray(value_array)
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 10), f"{key}", fill=(255, 255, 255))
                image.append(np.array(pil_img))
            frame = np.concatenate(image, axis=1)
        elif isinstance(img, Image.Image):
            frame = np.array(img.convert("RGB"))
        else:
            frame = np.array(img)
        video_writer.append_data(frame)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_prediction_video(
    rollout_dir,
    gt_frames,
    pred_frames,
    idx,
    replan_idx,
    success,
    task_description,
    log_file=None,
    fps=8,
):
    """Saves an MP4 comparison of ground-truth and predicted future frames for one replanning clip."""
    num_frames = min(len(gt_frames), len(pred_frames))
    if num_frames <= 0:
        raise ValueError("Cannot save prediction video with empty GT/pred frame lists.")

    stitched_frames = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        if isinstance(gt_frame, dict):
            gt_images = []
            for value in gt_frame.values():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                gt_images.append(value_array)
            gt_image = np.concatenate(gt_images, axis=1)
        elif isinstance(gt_frame, Image.Image):
            gt_image = np.array(gt_frame.convert("RGB"))
        else:
            gt_image = np.array(gt_frame)

        if isinstance(pred_frame, Image.Image):
            pred_image = np.array(pred_frame.convert("RGB"))
        else:
            pred_image = np.array(pred_frame)

        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_pil = Image.fromarray(gt_image)
        ImageDraw.Draw(gt_pil).text((10, 10), "gt", fill=(255, 255, 255))
        pred_pil = Image.fromarray(pred_image)
        ImageDraw.Draw(pred_pil).text((10, 10), "pred", fill=(255, 255, 255))
        stitched_frames.append(
            Image.fromarray(np.concatenate([np.array(pred_pil), np.array(gt_pil)], axis=0))
        )

    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    try:
        replan_tag = f"{int(replan_idx):04d}"
    except (TypeError, ValueError):
        replan_tag = str(replan_idx)
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}"
        f"--task={processed_task_description}--replan={replan_tag}--gt-pred.mp4"
    )
    save_mp4(stitched_frames, mp4_path, fps=fps)
    print(f"Saved predicted future comparison MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved predicted future comparison MP4 at path {mp4_path}\n")
    return mp4_path

def binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = (v > 0.5)
    return np.asarray(bin_val, dtype=np.float32)


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
