#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextvars
import glob
import importlib
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

import av
import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
from PIL import Image


# Per-call profile dict shared with LeRobotDataset.__getitem__ via ContextVar.
# When non-None, fine-grained timings (codec open/decode, etc.) are written into
# the dict so the caller (BaseLerobotDataset) can surface them under sample["_profile"].
_PROFILE_CTX: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "_lerobot_profile_ctx", default=None
)


def _profile_add(key: str, dt: float) -> None:
    """Accumulate `dt` into the active profile dict under `key` if profiling is on."""
    profile = _PROFILE_CTX.get()
    if profile is None:
        return
    profile[key] = profile.get(key, 0.0) + dt


def get_safe_default_codec():
    override = os.environ.get("JAM_VIDEO_BACKEND") or os.environ.get("LEROBOT_VIDEO_BACKEND")
    if override:
        override = override.strip().lower()
        if override not in {"auto", "default"}:
            if override not in {"torchcodec", "pyav", "video_reader"}:
                raise ValueError(
                    "Unsupported video backend override "
                    f"{override!r}. Expected one of: auto, torchcodec, pyav, video_reader."
                )
            logging.info("Using video backend override: %s", override)
            return override

    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..

    Returns:
        torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        try:
            return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
        except Exception as err:
            if isinstance(err, RuntimeError) and "no more frames left to decode" in str(err):
                warnings.warn(
                    "torchcodec hit video EOF while decoding; falling back to torchvision/pyav "
                    f"for {video_path}."
                )
                return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend="pyav")
            raise
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    profile_on = _PROFILE_CTX.get() is not None
    _t0 = time.perf_counter() if profile_on else 0.0

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    reader = torchvision.io.VideoReader(video_path, "video")
    if backend == "pyav":
        # torchvision's PyAV VideoReader accepts ``num_threads`` but does not
        # apply it to the codec context in torchvision 0.22.  When explicitly
        # requested by a launcher, configure the underlying decoder before the
        # first packet is read.  One thread avoids repeatedly creating an
        # automatic libdav1d decoder pool in every DataLoader worker.  Decoder
        # threading only changes scheduling: requested timestamps and decoded
        # pixels are unchanged.
        pyav_threads_raw = os.environ.get("JAM_PYAV_THREADS")
        if pyav_threads_raw is not None:
            try:
                pyav_threads = int(pyav_threads_raw)
            except ValueError as err:
                reader.container.close()
                raise ValueError(
                    "JAM_PYAV_THREADS must be a non-negative integer; "
                    f"got {pyav_threads_raw!r}."
                ) from err
            if pyav_threads < 0:
                reader.container.close()
                raise ValueError(
                    "JAM_PYAV_THREADS must be a non-negative integer; "
                    f"got {pyav_threads}."
                )
            pyav_stream = reader.container.streams.get(**reader.pyav_stream)[0]
            pyav_stream.codec_context.thread_count = pyav_threads
    if profile_on:
        _t1 = time.perf_counter()
        _profile_add("codec.open", _t1 - _t0)
        _t0 = _t1

    # RoboTwin stores a key frame every two frames.  For endpoint-only
    # training, the generic torchvision path nevertheless seeks once at t and
    # then decodes every frame through t+16.  When explicitly enabled, seek to
    # each requested endpoint independently in the *same* PyAV container.  We
    # retain the exact nearest-timestamp selection and tolerance check below;
    # only discarded intermediate frames are skipped.  This is opt-in because
    # densely sampled clips are better served by the original contiguous
    # decoder loop.
    sparse_endpoint_seek = (
        backend == "pyav"
        and os.environ.get("JAM_PYAV_SPARSE_ENDPOINT_SEEK", "0") == "1"
        and len(timestamps) > 1
    )
    if sparse_endpoint_seek:
        closest_frames = []
        closest_ts_values = []
        frames_decoded = 0
        try:
            for query_timestamp in timestamps:
                reader.seek(query_timestamp, keyframes_only=True)
                # Iterate over raw PyAV frames so discarded frames between the
                # preceding keyframe and the endpoint do not pay an RGB
                # conversion and Tensor allocation.  The selected frame is
                # converted with the exact operations used by torchvision's
                # PyAV VideoReader.__next__, preserving pixels bit-for-bit.
                best_frame = None
                best_ts = None
                best_distance = None
                iterator = reader._c
                for frame in iterator:
                    current_ts = float(frame.pts * frame.time_base)
                    if log_loaded_timestamps:
                        logging.info(f"frame loaded at timestamp={current_ts:.4f}")
                    frames_decoded += 1
                    distance = abs(float(query_timestamp) - current_ts)
                    if best_distance is None or distance < best_distance:
                        best_frame = frame
                        best_ts = current_ts
                        best_distance = distance
                    if current_ts >= query_timestamp:
                        break
                # Explicitly release the active generator before the next
                # seek.  This avoids carrying a stale PyAV packet iterator
                # across seeks in a long-lived DataLoader worker.
                del iterator
                if best_frame is None or best_ts is None or best_distance is None:
                    raise RuntimeError(
                        f"No frame decoded after seeking to timestamp={query_timestamp} "
                        f"in {video_path}."
                    )
                if not best_distance < tolerance_s:
                    raise AssertionError(
                        "A requested timestamp unexpectedly violates the tolerance "
                        f"({best_distance} > tolerance_s={tolerance_s}).\n"
                        f"queried timestamp: {query_timestamp}\n"
                        f"closest timestamp: {best_ts}\n"
                        f"video: {video_path}\nbackend: {backend}"
                    )
                closest_frames.append(
                    torch.as_tensor(best_frame.to_rgb().to_ndarray()).permute(2, 0, 1)
                )
                closest_ts_values.append(best_ts)
        finally:
            reader.container.close()
        if profile_on:
            _t1 = time.perf_counter()
            _profile_add("codec.decode", _t1 - _t0)
            _profile_add("codec.frames_decoded", float(frames_decoded))
        closest_frames = torch.stack(closest_frames).type(torch.float32) / 255
        if log_loaded_timestamps:
            logging.info(f"closest_ts={torch.tensor(closest_ts_values)}")
        assert len(timestamps) == len(closest_frames)
        return closest_frames

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    if profile_on:
        _t1 = time.perf_counter()
        _profile_add("codec.decode", _t1 - _t0)
        _profile_add("codec.frames_decoded", float(len(loaded_frames)))
        _t0 = _t1

    # Use float32 for timestamp distance computation (torch.cdist doesn't support bfloat16).
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    device: str = "cpu",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """

    if importlib.util.find_spec("torchcodec"):
        from torchcodec.decoders import VideoDecoder
    else:
        raise ImportError("torchcodec is required but not available.")

    profile_on = _PROFILE_CTX.get() is not None
    _t0 = time.perf_counter() if profile_on else 0.0
    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    if profile_on:
        _t1 = time.perf_counter()
        _profile_add("codec.open", _t1 - _t0)
        _t0 = _t1
    loaded_frames = []
    loaded_ts = []
    metadata = decoder.metadata
    average_fps = metadata.average_fps

    frame_indices = [round(ts * average_fps) for ts in timestamps]

    frames_batch = decoder.get_frames_at(indices=frame_indices)
    if profile_on:
        _t1 = time.perf_counter()
        _profile_add("codec.decode", _t1 - _t0)
        _profile_add("codec.frames_decoded", float(len(frame_indices)))
        _t0 = _t1

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logging.info(f"Frame loaded at timestamp={pts:.4f}")

    # Use float32 for timestamp distance computation (torch.cdist doesn't support bfloat16).
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to float32 in [0,1] range (channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames

def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    # Check encoder availability
    if vcodec == "h264_nvenc" :
        return encode_video_frames_ffmpeg(
            imgs_dir, video_path, fps, pix_fmt = pix_fmt, overwrite=overwrite
        )

    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        logging.warning(
            f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    # Get input frames
    template = "frame_" + ("[0-9]" * 6) + ".jpeg"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    # Define video output frame size (assuming all input frames are the same size)
    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    dummy_image = Image.open(input_list[0])
    width, height = dummy_image.size

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_data in input_list:
            input_image = Image.open(input_data).convert("RGB")
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")

# GPU speed up video encoding
def encode_video_frames_ffmpeg(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "h264_nvenc",
    pix_fmt: str = "yuv420p",
    g: int | None = 4,
    crf: int | None = 30,
    overwrite: bool = False,
) -> None:
    import subprocess
    import shutil
    
    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)
    
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not installed or not in system PATH")
    
    # Build ffmpeg command
    cmd = [
        ffmpeg_path,
        "-y" if overwrite else "-n",
        "-framerate", str(fps),
        "-pattern_type", "sequence",
        "-i", str(imgs_dir / "frame_%06d.jpeg"),
        "-vcodec", vcodec,
        "-pix_fmt", pix_fmt,
    ]
    
    if g is not None:
        cmd.extend(["-g", str(g)])
        
    if crf is not None:
        cmd.extend(["-crf", str(crf)])
        
    cmd.append(str(video_path))
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg failed:\nCommand: {' '.join(cmd)}\nStdout: {e.stdout}\nStderr: {e.stderr}"
        logging.error(error_msg)
        logging.error(video_path)
        logging.error(vcodec)
        print(f"ffmpeg {' '.join(cmd)}")
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Failed to run FFmpeg command: {' '.join(cmd)}\nError: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting audio stream information
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # In an ideal loseless case : bit depth x sample rate x channels = bit rate.
        # In an actual compressed case, the bit rate is set according to the compression level : the lower the bit rate, the more compression is applied.
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # Number of samples per second
        # In an ideal loseless case : fixed number of bits per sample.
        # In an actual compressed case : variable number of bits per sample (often reduced to match a given depth rate).
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # Reset logging level
    av.logging.restore_default_callback()

    return audio_info


def get_video_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting video stream information
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt
        video_info["video.is_depth_map"] = False

        # Calculate fps from r_frame_rate
        video_info["video.fps"] = int(video_stream.base_rate)

        pixel_channels = get_video_pixel_channels(video_stream.pix_fmt)
        video_info["video.channels"] = pixel_channels

    # Reset logging level
    av.logging.restore_default_callback()

    # Adding audio stream information
    video_info.update(**get_audio_info(video_path))

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # Grayscale
    elif image.mode == "LA":
        return 2  # Grayscale + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("Unknown format")
