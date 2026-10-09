import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fsync_dir(path) -> None:
    """Fsync a directory so a rename inside it survives a crash."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_parquet(df: pd.DataFrame, path) -> None:
    """Write `df` to `path` via a temporary file + fsync + os.replace, so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def read_parquet_dir(directory, pattern: str = "*.parquet") -> pd.DataFrame:
    """Concatenate every parquet part file in `directory` (recursively). Returns an empty frame if there are none."""
    parts = sorted(Path(directory).rglob(pattern)) if Path(directory).exists() else []
    frames = [pd.read_parquet(p) for p in parts]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def setup_logging(log_dir, name: str, level: str = "INFO") -> None:
    """Log to stderr and to `<log_dir>/<name>_<timestamp>.log`."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "{time:YYYY-MM-DD HH:mm:ss,SSS} {level} {name}: {message}"
    logger.remove()
    logger.add(sys.stderr, level=level, format=fmt)
    logger.add(log_dir / f"{name}_{time.strftime('%Y%m%dT%H%M%S')}.log", level=level, format=fmt)
