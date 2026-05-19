"""
DiskCache — lightweight dual-layer cache: in-memory LRU dict + JSON files on disk.

Usage:
    cache = DiskCache(cache_dir, max_memory=100)
    cache.set(key, value)          # store string value
    v = cache.get(key)             # returns str | None
    cache.clear()                  # wipe memory + disk (called on build_index)
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _sha(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:24]


class DiskCache:
    """
    Dual-layer cache: hot in-memory LRU dict + cold JSON files on disk.

    The on-disk format is simple: one file per entry named <hash>.json,
    containing {"k": "<original key>", "v": "<value>"}.  Values must be
    plain strings (callers JSON-encode complex objects before storing).

    Thread-safety: reads are safe; concurrent writes may occasionally write
    the same file twice, which is harmless (last writer wins, same value).
    """

    def __init__(self, cache_dir: Path, max_memory: int = 100) -> None:
        self._dir = cache_dir
        self._max = max_memory
        self._mem: dict[str, str] = {}  # hash -> value (ordered insertion = LRU approx)

    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[str]:
        h = _sha(key)
        if h in self._mem:
            return self._mem[h]
        p = self._dir / f"{h}.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("k") == key:
                    self._mem_put(h, data["v"])
                    return data["v"]
            except Exception:
                pass
        return None

    def set(self, key: str, value: str) -> None:
        h = _sha(key)
        self._mem_put(h, value)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            p = self._dir / f"{h}.json"
            p.write_text(json.dumps({"k": key, "v": value}, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.debug("DiskCache write error: %s", exc)

    def clear(self) -> None:
        self._mem.clear()
        if self._dir.exists():
            try:
                shutil.rmtree(self._dir)
                logger.debug("DiskCache cleared: %s", self._dir)
            except Exception as exc:
                logger.debug("DiskCache clear error: %s", exc)

    # ------------------------------------------------------------------
    def _mem_put(self, h: str, value: str) -> None:
        if h in self._mem:
            del self._mem[h]
        self._mem[h] = value
        if len(self._mem) > self._max:
            # Evict the oldest entry (first key in insertion order)
            self._mem.pop(next(iter(self._mem)))
