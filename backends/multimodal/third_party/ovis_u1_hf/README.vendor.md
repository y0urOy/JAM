# Vendored Ovis-U1 code and configuration

This directory holds the modeling code, configuration and tokenizer metadata of the Hugging Face repository
`AIDC-AI/Ovis-U1-3B` at revision `cbc005dd`. The Ovis-U1 and DINOv2 recipes import it as `ovis_u1_hf`.

- Source repository: https://huggingface.co/AIDC-AI/Ovis-U1-3B
- File list: [`MANIFEST.json`](./MANIFEST.json)
- Weights: `python tools/download_assets.py ovis`

## Local changes

Carried over from ImageWAM: `configuration_ovis_u1.py` and `modeling_ovis_u1.py` register their Auto classes only
once, so the modules can be imported more than once, and `modeling_yak.py` imports `diffusers` only when the full
Yak image generator is built.

Added for Transformers 5, under which JAM loads `ovis_u1_hf.modeling_ovis_u1.OvisU1` directly, so that the model
keeps the numerics it had under Transformers 4.51.3: `modeling_ovis_u1.py` keeps the Qwen3 language model on eager
attention, loads the text tokenizer without the remote-code prompt, calls `post_init()` and accepts the new
`tie_weights` arguments; `modeling_aimv2.py` recomputes the rotary `inv_freq` buffer when weights are initialized.

## License

These files keep the upstream Apache-2.0 license in [`LICENSE`](./LICENSE) and the upstream [`Notice`](./Notice).
