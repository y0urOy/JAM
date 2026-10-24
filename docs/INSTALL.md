# Installation

Use Linux and Python 3.10. The recorded policy runtime used PyTorch 2.7.1, torchvision 0.22.1 and CUDA 12.8. The installer creates a new venv and refuses to overwrite an existing one.

| Profile | Transformers | DeepSpeed | Backends |
|---|---:|---:|---|
| `wan` | 4.49.0 | 0.18.5 | JAM (Wan) |
| `ovis` | 4.51.3 | 0.19.2 | JAM (Ovis-U1) and DINOv2 |
| `flux` | 4.56.1 | 0.18.5 | FLUX.2; Qwen3 text caches for FLUX.2 and DINOv2 |
| `qwen` | 5.13.1 | 0.18.5 | Native Qwen3.5 action expert and shared-weight ablations |

```bash
python3.10 tools/bootstrap.py ovis --venv .venv-ovis --libero --dry-run
python3.10 tools/bootstrap.py ovis --venv .venv-ovis --libero
source .venv-ovis/bin/activate
python -m pip check
python tools/jam.py compose ovis-jam
```

The dependency files in `environments/` pin the training libraries. `observed_*.json` records the full installed-package versions in the original environments without local paths or credentials. System libraries, drivers and GPU architecture are separate requirements. A clean CUDA installation on a new machine has not been validated. If the selected wheel index has no wheel for the host architecture, use the matching NVIDIA/PyTorch distribution and record that change; do not silently mix CUDA/PyTorch versions.

Qwen may optionally use the recorded fused kernels:

```bash
python3.10 tools/bootstrap.py qwen --venv .venv-qwen --fast-kernels --libero
```

Building `causal-conv1d` requires a compatible CUDA compiler. Without the optional fused libraries, the Transformers fallback may be slower. Keep Qwen's Transformers-5 environment separate from Wan/Ovis.

## Simulators

```bash
# Use the active policy venv; each checkout is pinned and installed without changing dependencies.
python tools/setup_simulators.py libero
python tools/setup_simulators.py plus
```

The evaluators select the corresponding checkout through `PYTHONPATH`. Use MuJoCo 3.3.2, robosuite 1.4.0, and BDDL 1.0.1. Download benchmark assets and initial states from the upstream projects. Some machines require local EGL/NVIDIA system libraries; `EVALUATION.env_subprocess=true` keeps MuJoCo/EGL outside the CUDA policy process.

`third_party/RoboTwin/` is a trimmed copy of RoboTwin 2.0 ([RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) at `bf44be5`, MIT) with nine evaluation files modified. Its installer and asset scripts need files that are not copied here: install the environment and download the separately licensed assets from an upstream checkout at the same commit, then point the evaluator at those assets. Rendering on a new GPU has not been validated.
