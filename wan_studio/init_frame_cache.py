from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class CacheItem:
    path: str
    seed: int = 0
    created_ts: int = 0


def load_cache(path: str) -> Dict[str, CacheItem]:
    try:
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: Dict[str, CacheItem] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if not isinstance(v, dict):
                    continue
                p = v.get("path")
                if not p:
                    continue
                out[str(k)] = CacheItem(
                    path=str(p),
                    seed=int(v.get("seed", 0) or 0),
                    created_ts=int(v.get("created_ts", 0) or 0),
                )
        return out
    except Exception:
        return {}


def save_cache(path: str, cache: Dict[str, CacheItem]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    data = {k: asdict(v) for k, v in (cache or {}).items()}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
