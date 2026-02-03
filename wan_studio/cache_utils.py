from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


def _log(log: Optional[Callable[[str], None]], msg: str) -> None:
    if callable(log):
        try:
            log(msg)
        except Exception:
            pass


def dir_size_bytes(path: str) -> int:
    """Return total size of files under path (bytes)."""
    total = 0
    if not path or not os.path.exists(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return int(total)


def clear_dir(path: str) -> None:
    """Delete all children of a directory (keeps the directory itself)."""
    if not path or not os.path.isdir(path):
        return
    for name in os.listdir(path):
        p = os.path.join(path, name)
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except Exception:
            pass


@dataclass
class CacheEntryInfo:
    path: str
    size_bytes: int
    mtime: float


def list_entries(root: str) -> List[CacheEntryInfo]:
    out: List[CacheEntryInfo] = []
    if not root or not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.exists(p):
            continue
        try:
            st = os.stat(p)
            mtime = float(st.st_mtime)
        except Exception:
            mtime = 0.0
        try:
            size = dir_size_bytes(p) if os.path.isdir(p) else int(os.path.getsize(p))
        except Exception:
            size = 0
        out.append(CacheEntryInfo(path=p, size_bytes=int(size), mtime=mtime))
    # Oldest first
    out.sort(key=lambda e: (e.mtime, e.path))
    return out


def prune_to_quota(root: str, max_bytes: int, *, log: Optional[Callable[[str], None]] = None, keep_min: int = 0) -> Tuple[int, int]:
    """Prune entries under root until directory size <= max_bytes.

    Returns: (deleted_entries, deleted_bytes)

    Strategy:
      - Compute total size.
      - Delete oldest entries first (by mtime).
      - Never delete below keep_min entries.
    """
    if not root or max_bytes <= 0 or not os.path.isdir(root):
        return (0, 0)

    entries = list_entries(root)
    if keep_min and len(entries) <= keep_min:
        return (0, 0)

    total = sum(e.size_bytes for e in entries)
    if total <= max_bytes:
        return (0, 0)

    deleted_e = 0
    deleted_b = 0

    _log(log, f"[Cache] Pruning {root} (size={total/1e9:.2f}GB > limit={max_bytes/1e9:.2f}GB)")

    for e in entries:
        if total <= max_bytes:
            break
        # keep minimal number of newest entries
        if keep_min and (len(entries) - deleted_e) <= keep_min:
            break
        try:
            if os.path.isdir(e.path) and not os.path.islink(e.path):
                shutil.rmtree(e.path)
            else:
                os.remove(e.path)
            total -= int(e.size_bytes)
            deleted_e += 1
            deleted_b += int(e.size_bytes)
        except Exception:
            continue

    if deleted_e:
        _log(log, f"[Cache] Pruned: {deleted_e} entries, freed {deleted_b/1e9:.2f}GB")

    return (deleted_e, deleted_b)
