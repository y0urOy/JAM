# Installation

Use Linux and Python 3.10. The recorded policy runtime used PyTorch 2.7.1, torchvision 0.22.1 and CUDA 12.8. The ten recipes share one environment. The installer creates a new venv and refuses to overwrite an existing one.

| Library | Version | Used by |
|---|---:|---|
| Transformers | 5.13.1 | all backbones |
| DeepSpeed | 0.19.2 | training |
| diffusers | 0.34.0 | Ovis-U1 and DINOv2 (the Yak generator of Ovis-U1) |
| flash-attn | 2.8.3.post1 | Ovis-U1 and DINOv2 (the AIMv2 vision encoder of Ovis-U1) |
| fla-core, causal-conv1d | 0.5.1, 1.6.2.post1 | Qwen3.5-2B linear attention |

```bash
python3.10 tools/bootstrap.py --venv .venv --libero --dry-run
python3.10 tools/bootstrap.py --venv .venv --libero
source .venv/bin/activate
python -m pip check
python tools/jam.py compose libero-ovis-jam
```

`environments/requirements.txt` lists the packages that JAM needs directly, and `environments/constraints.txt` pins every installed version, including transitive dependencies. `environments/observed.json` records the installed-package versions of the validated runtime, without local paths or credentials. System libraries, drivers and GPU architecture are separate requirements. If the selected wheel index has no wheel for the host architecture, use the matching NVIDIA/PyTorch distribution and record that change; do not silently mix CUDA/PyTorch versions.

The bootstrap installs PyTorch, the pinned wheels, both backends and, with `--libero`, the LIBERO runtime. It then builds DeepSpeed, flash-attn and causal-conv1d against the installed PyTorch with `--no-build-isolation`; `DS_BUILD_OPS=0` leaves DeepSpeed's ops to its just-in-time compiler. The Qwen3.5 recipes need fla-core and causal-conv1d: the Transformers fallback kernels compute different numbers and take about twice the memory.

flash-attn and causal-conv1d need `nvcc` from a CUDA 12.8 toolkit (on `PATH` or in `CUDA_HOME`) and a host compiler that it supports. Their setup scripts download a matching prebuilt wheel from the projects' GitHub releases when one exists and compile from source otherwise. On aarch64 the bootstrap sets `FLASH_ATTENTION_FORCE_BUILD=TRUE` and `CAUSAL_CONV1D_FORCE_BUILD=TRUE`, so both are compiled from source, as in the recorded environment. flash-attn compiles kernels for compute capabilities 8.0, 9.0, 10.0 and 12.0 by default; the recorded build has only 9.0 (H100/GH200), which `FLASH_ATTN_CUDA_ARCHS=90` selects. `MAX_JOBS` sets the number of parallel compile jobs, each of which can take several GB of memory. For example, on a GH200 node:

```bash
module load gcc/12.4.0 cuda/12.8    # site-specific; provides gcc and nvcc
FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=16 python3.10 tools/bootstrap.py --venv .venv --libero
```

With these settings, the bootstrap took about 20 minutes on one GH200 node with 64 CPU cores, of which 11 minutes compiled flash-attn and 6 compiled causal-conv1d.

## Simulators

```bash
# Run in the policy venv; checkouts are pinned and patched, and the evaluators import them through PYTHONPATH.
python tools/setup_simulators.py libero
python tools/download_assets.py libero-plus-assets
python tools/setup_simulators.py plus
```

`setup_simulators.py` applies the two compatibility patches in `patches/` (LIBERO's initial-state files load with `weights_only=False` under PyTorch 2.7; LIBERO-Plus accepts path objects), unpacks the LIBERO-Plus assets into its checkout (LIBERO keeps its assets in git) and writes each checkout's LIBERO configuration to `third_party/.libero/` or `third_party/.libero_plus/`. On an existing checkout at the pinned revision, it applies a missing patch and writes the configuration. The evaluators select the checkout through `PYTHONPATH` and its configuration through `LIBERO_CONFIG_PATH`. Use MuJoCo 3.3.2, robosuite 1.4.0, and BDDL 1.0.1 (`--libero` of the bootstrap). LIBERO-Plus also imports Wand, which needs the system ImageMagick library (`libmagickwand`). Some machines require local EGL/NVIDIA system libraries; `EVALUATION.env_subprocess=true` keeps MuJoCo/EGL outside the CUDA policy process.

The downloader and setup verify archive sizes and SHA256 values from `reproduction/assets.json`. Setup checks the recorded simulator files and extracts each missing asset directory atomically, so interrupted extraction can be retried.

`third_party/RoboTwin/` is a trimmed copy of RoboTwin 2.0 ([RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) at `bf44be5`, MIT) with nine evaluation files modified. The simulator runs in the policy's own environment (the `--venv` given to `tools/jam.py eval`) with `third_party/RoboTwin` as its root. RoboTwin 2.0 success rates depend on the simulator version: use SAPIEN 3.0.0b1, as in the paper. SAPIEN 3.0.3 gives success rates that are not comparable, and the evaluator refuses versions other than 3.0.0b1 unless `JAM_ALLOW_SAPIEN_VERSION` names the installed one. Install the simulator packages into the policy environment under its constraints. The shared setuptools 81 pin retains `pkg_resources`, which SAPIEN 3.0.0b1 imports:

```bash
python -m pip install sapien==3.0.0b1 mplib==0.2.1 -c environments/constraints.txt
python -m pip install -r environments/robotwin.txt -c environments/constraints.txt
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
git clone --branch v0.7.8 https://github.com/NVlabs/curobo.git third_party/RoboTwin/envs/curobo
python -m pip install --no-build-isolation -e third_party/RoboTwin/envs/curobo
python tools/download_assets.py robotwin-assets
python tools/setup_simulators.py robotwin
```

Do not run RoboTwin's own `script/requirements.txt` or `script/_install.sh` in a policy environment; they install PyTorch 2.4.1 and other conflicting versions. SAPIEN 3.0.0b1 and MPlib 0.2.1 publish x86_64 wheels only. On aarch64, build SAPIEN 3.0.0b1 (which needs aarch64 build fixes) and MPlib v0.2.1 from source in place of the first command; a source build of MPlib v0.2.1 reports version 0.2.0. The last command checks the SAPIEN version, unpacks the pinned RoboTwin assets into `third_party/RoboTwin/assets/`, applies the SAPIEN and MPlib source edits that RoboTwin's `script/_install.sh` describes, and points the embodiment configurations at the assets (`script/update_embodiment_config_path.py`).

`toppra==0.6.3` builds from source on both architectures, and Zarr's `numcodecs` dependency also builds from source on aarch64. Provide a C/C++ compiler and Python development headers; pip installs the Cython and NumPy build requirements.
