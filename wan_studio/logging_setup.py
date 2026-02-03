from __future__ import annotations

import logging
import os
from typing import Iterable


class _SuppressWanLoraPrefixWarning(logging.Filter):
    """Silence a noisy diffusers log message seen with some Wan2.x LoRA files.

    Some diffusers versions may emit a WARNING like:
      "No LoRA keys associated to WanTransformer3DModel found with the prefix='transformer'"

    It's usually harmless (LoRA state dict doesn't target that submodule), but it
    confuses users because it looks like an error. We filter it here.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 (shadowing built-in)
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "No LoRA keys associated to WanTransformer3DModel" in msg and "prefix='transformer'" in msg:
            return False
        return True


def install_log_filters(logger_names: Iterable[str] | None = None) -> None:
    """Install log filters to hide known-noisy messages.

    Call this as early as possible (before importing/creating diffusers pipelines)
    so the filter catches messages during model/LoRA loading.
    """

    names = tuple(logger_names) if logger_names else (
        "diffusers",
        "diffusers.loaders",
        "diffusers.loaders.lora",
        "diffusers.utils",
    )
    flt = _SuppressWanLoraPrefixWarning()
    for n in names:
        try:
            logging.getLogger(n).addFilter(flt)
        except Exception:
            pass


def install_env_defaults() -> None:
    """Set safe default env vars (only if not already set by user)."""

    # Common source of "ghost" bugs: running code from folder A with a venv
    # created in folder B (old build, mismatched deps, wrong tools paths).
    # This doesn't crash, but it creates very confusing behavior, so we warn early.
    try:
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        exe = Path(sys.executable).resolve()
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            venv_path = Path(venv).resolve()
            # If the venv isn't inside the current repo root, warn.
            if repo_root not in venv_path.parents:
                sys.stderr.write(
                    "[Env] ⚠️ Tu exécutes Wan Studio avec un venv situé ailleurs que ce dossier.\n"
                    f"      code: {repo_root}\n"
                    f"      venv: {venv_path} (python={exe})\n"
                    "      → Recommandé: recréer un .venv dans ce dossier pour éviter des versions mélangées.\n"
                )
    except Exception:
        pass

    # Reduce allocator fragmentation for long I2V jobs.
    # (Some torch builds still use PYTORCH_CUDA_ALLOC_CONF; newer prefer PYTORCH_ALLOC_CONF)
    conf = "expandable_segments:True,max_split_size_mb:128"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", conf)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", conf)
