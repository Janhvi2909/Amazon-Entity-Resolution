"""Checkpointing that trusts nothing it has not opened.

Every artifact is written to ``<path>.tmp``, fsynced, renamed, and only then gets a marker
``<path>.ok`` holding its row count and byte size. ``good(path)`` re-opens the file (parquet
footer, numpy header, LightGBM model text) and checks both against the marker, so a file
torn by a killed job, or one without a marker, counts as absent and is recomputed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl


def _marker(path: Path) -> Path:
    return path.with_name(path.name + ".ok")


def _fsync_rename(tmp: Path, path: Path) -> None:
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _mark(path: Path, rows: int) -> None:
    m = _marker(path)
    tmp = m.with_name(m.name + ".tmp")
    tmp.write_text(json.dumps({"rows": int(rows), "bytes": path.stat().st_size}))
    _fsync_rename(tmp, m)


def _rows(path: Path) -> int:
    s = path.suffix
    if s == ".parquet":
        return int(pl.scan_parquet(path).select(pl.len()).collect().item())  # reads the footer
    if s == ".npy":
        return int(np.load(path, mmap_mode="r").shape[0])
    if s == ".txt":  # LightGBM model: must end with its closing section
        with open(path, "rb") as f:
            f.seek(max(0, path.stat().st_size - 4096))
            tail = f.read()
        if b"end of parameters" not in tail and b"pandas_categorical" not in tail:
            raise ValueError("model file truncated")
        return 1
    if s == ".json":
        json.loads(path.read_text())
        return 1
    if s == ".tsv":
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    raise ValueError(f"no reader for {path}")


def good(path: str | Path, min_rows: int = 0) -> bool:
    """True only if ``path`` opens, its marker exists and both agree on rows and size."""
    path = Path(path)
    m = _marker(path)
    try:
        meta = json.loads(m.read_text())
        if path.stat().st_size != meta["bytes"]:
            return False
        n = _rows(path)
        return n == meta["rows"] and n >= min_rows
    except Exception:
        return False


def write_parquet(df: pl.DataFrame, path: str | Path, allow_empty: bool = False) -> Path:
    path = Path(path)
    if df.height == 0 and not allow_empty:
        raise RuntimeError(f"refusing to write an empty artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    _fsync_rename(tmp, path)
    _mark(path, df.height)
    return path


def write_npy(a: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    if len(a) == 0:
        raise RuntimeError(f"refusing to write an empty artifact: {path}")
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, a)
    _fsync_rename(tmp, path)
    _mark(path, len(a))
    return path


def write_json(d: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, default=float))
    _fsync_rename(tmp, path)
    _mark(path, 1)
    return path


def save_model(model, path: str | Path) -> Path:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.txt")
    model.save_model(str(tmp))
    _fsync_rename(tmp, path)
    _mark(path, 1)
    return path


def mark_existing(path: str | Path) -> Path:
    """Marker for a file written by another writer (e.g. a TSV), after re-reading it."""
    path = Path(path)
    _mark(path, _rows(path))
    return path


def clean_dir(d: str | Path, pattern: str) -> tuple[list[Path], list[Path]]:
    """Delete every ``pattern`` file in ``d`` that is not good (and stray .tmp files). Returns (kept, deleted)."""
    d = Path(d)
    kept, deleted = [], []
    if not d.exists():
        return kept, deleted
    for tmp in d.glob("*.tmp*"):
        tmp.unlink(missing_ok=True)
    for p in sorted(d.glob(pattern)):
        if p.name.endswith(".ok"):
            continue
        if good(p):
            kept.append(p)
        else:
            p.unlink(missing_ok=True)
            _marker(p).unlink(missing_ok=True)
            deleted.append(p)
    for m in d.glob(pattern + ".ok"):  # markers whose file is gone
        if not Path(str(m)[:-3]).exists():
            m.unlink(missing_ok=True)
    return kept, deleted


def require(n: int, what: str, minimum: int = 1) -> int:
    """Fail loudly on empty or implausibly small results."""
    if n < minimum:
        raise RuntimeError(f"FATAL: {what} has {n:,} rows (expected >= {minimum:,})")
    return n
