from __future__ import annotations

import sys
from pathlib import Path


def _patch_transformers_auto_register_exist_ok() -> None:
    """Allow Ovis remote code to re-register configs already present upstream."""
    try:
        from transformers import AutoConfig, AutoModel
    except Exception:
        return

    if not getattr(AutoConfig.register, "_imagewam_exist_ok_patch", False):
        original_config_register = AutoConfig.register

        def register_config_exist_ok(model_type, config, **kwargs):
            try:
                return original_config_register(model_type, config, **kwargs)
            except ValueError as exc:
                if "already" not in str(exc):
                    raise
                kwargs["exist_ok"] = True
                return original_config_register(model_type, config, **kwargs)

        register_config_exist_ok._imagewam_exist_ok_patch = True
        AutoConfig.register = register_config_exist_ok

    if not getattr(AutoModel.register, "_imagewam_exist_ok_patch", False):
        original_model_register = AutoModel.register

        def register_model_exist_ok(config_class, model_class, **kwargs):
            try:
                return original_model_register(config_class, model_class, **kwargs)
            except ValueError as exc:
                if "already" not in str(exc):
                    raise
                kwargs["exist_ok"] = True
                return original_model_register(config_class, model_class, **kwargs)

        register_model_exist_ok._imagewam_exist_ok_patch = True
        AutoModel.register = register_model_exist_ok


def ensure_ovis_u1_remote_code_importable() -> None:
    """Expose vendored Ovis-U1 HF remote code as `ovis_u1_hf`."""
    _patch_transformers_auto_register_exist_ok()
    repo_root = Path(__file__).resolve().parents[4]
    third_party = repo_root / "third_party"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
