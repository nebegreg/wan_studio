from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


def set_allocator_defaults() -> None:
    """Set safe allocator defaults (only if user didn't set them).

    PyTorch has multiple env vars across versions:
      - PYTORCH_CUDA_ALLOC_CONF (older)
      - PYTORCH_ALLOC_CONF (newer)
    We set BOTH (if absent) to enable expandable segments, which reduces
    fragmentation and improves long renders.
    """

    # Keep it small / conservative; users can override.
    conf = "expandable_segments:True,max_split_size_mb:128"

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", conf)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", conf)


def cuda_mem_info() -> Tuple[float, float]:
    """Return (free_gb, total_gb)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        free_b, total_b = torch.cuda.mem_get_info()
        return float(free_b) / (1024**3), float(total_b) / (1024**3)
    except Exception:
        return 0.0, 0.0


def cuda_cleanup(aggressive: bool = False) -> None:
    """Best-effort VRAM cleanup (safe to call often)."""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            if aggressive:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
    except Exception:
        pass


@dataclass
class _Resource:
    name: str
    obj: Any
    kind: str
    unload: Optional[Callable[[], None]] = None


class ResourceManager:
    """Tiny in-process manager for heavy GPU resources.

    Goal: prevent 'Flux2 eats VRAM → Wan OOM' by enforcing ordering:
      - unload init-image pipeline (Flux/SDXL) before loading I2V
      - do a consistent cuda_cleanup between phases

    This is *not* a perfect allocator, but it's robust and low-risk.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._resources: Dict[str, _Resource] = {}

    def register(self, name: str, obj: Any, *, kind: str, unload: Optional[Callable[[], None]] = None) -> None:
        with self._lock:
            self._resources[name] = _Resource(name=name, obj=obj, kind=kind, unload=unload)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._resources.pop(name, None)

    def unload(self, name: Optional[str] = None, *, kind: Optional[str] = None, aggressive: bool = False) -> None:
        """Unload one or many resources."""
        with self._lock:
            if name is not None:
                targets = [self._resources.get(name)]
            else:
                targets = list(self._resources.values())

        for r in targets:
            if r is None:
                continue
            if kind is not None and r.kind != kind:
                continue
            try:
                if r.unload is not None:
                    r.unload()
            except Exception:
                pass
            try:
                # Drop references
                r.obj = None
            except Exception:
                pass
            self.unregister(r.name)

        cuda_cleanup(aggressive=aggressive)

    def ensure_free(self, min_free_gb: float, *, log=None, reason: str = "", aggressive: bool = False) -> bool:
        """Try to ensure at least min_free_gb VRAM is free.

        Returns True if free VRAM >= min_free_gb after cleanup attempts.
        """
        free, total = cuda_mem_info()
        if free >= float(min_free_gb):
            return True

        # First pass: always cleanup caches.
        if log:
            try:
                log(f"[VRAM] low ({free:.2f}GB free / {total:.2f}GB total). cleanup… {reason}")
            except Exception:
                pass
        cuda_cleanup(aggressive=False)

        free, total = cuda_mem_info()
        if free >= float(min_free_gb):
            return True

        # Second pass: unload init pipelines first (flux2/sdxl), then others.
        # We keep ordering deterministic.
        self.unload(kind="init", aggressive=False)
        free, total = cuda_mem_info()
        if free >= float(min_free_gb):
            return True

        if aggressive:
            self.unload(kind="i2v", aggressive=True)
            free, total = cuda_mem_info()
            return free >= float(min_free_gb)

        return False


_RM: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    global _RM
    if _RM is None:
        set_allocator_defaults()
        _RM = ResourceManager()
    return _RM
