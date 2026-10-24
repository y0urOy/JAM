# Third-party notices

- FastWAM: https://github.com/yuantianyuan01/FastWAM — MIT; retained at `backends/wan/LICENSE`.
- ImageWAM: https://github.com/yuyangalin/ImageWAM (derived from upstream commit `2d9774b`, with local modifications) — MIT; retained at `backends/multimodal/LICENSE`.
- RoboTwin: https://github.com/RoboTwin-Platform/RoboTwin (trimmed copy of commit `bf44be5`, nine evaluation files modified) — source MIT text retained in `third_party/RoboTwin/LICENSE`. External robot meshes, simulators and assets retain their own terms.
- LeRobot: https://github.com/huggingface/lerobot — copied readers retain their Apache-2.0 headers. Full upstream notice is in `licenses/LeRobot-LICENSE.txt`.
- Ovis-U1: https://huggingface.co/AIDC-AI/Ovis-U1-3B — Apache-2.0 metadata and the upstream license retained alongside its small code/config/tokenizer subset. Weights are not bundled.
- FLUX.2 code: https://github.com/black-forest-labs/flux2/tree/50fe5162777813d869182b139e83b10743caef15 — Apache-2.0 license retained in `backends/multimodal/third_party/flux2/LICENSE.md`. This does not relicense external FLUX weights.
- PyTorch3D: https://github.com/facebookresearch/pytorch3d — the quaternion and rotation helpers in `backends/*/src/*/datasets/lerobot/{utils,transforms}/rotation.py` are adapted from PyTorch3D (BSD-3-Clause); its license is in `licenses/PyTorch3D-LICENSE.txt`.
- NVIDIA: `backends/*/src/*/utils/misc.py` carries an NVIDIA Apache-2.0 header; the license text is in `licenses/Apache-2.0.txt`.
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO — fetched separately at the recorded revision, with its upstream license.
- LIBERO-Plus: https://github.com/sylvestf/LIBERO-plus — fetched separately at the recorded revision. No upstream source/assets are republished in this release.

The MIT license at this repository root applies to JAM additions and does not supersede any retained third-party notices or the download/access terms of pretrained models and datasets.
