"""Config loading, crash-safe parquet writes and Loguru logging setup."""

import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger


def load_config(path: str | Path) -> dict[str, Any]:
    """Loads a YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed config as a nested dict.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def fsync_dir(path: str | Path) -> None:
    """Fsyncs a directory so a rename inside it survives a crash.

    Args:
        path: Directory to fsync.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Writes a frame to parquet so readers never see a partial file.

    The frame goes to a temporary file, which is fsynced and then moved into
    place with `os.replace`; the parent directory is fsynced afterwards.

    Args:
        df: Frame to write (the index is not stored).
        path: Destination parquet path. Missing parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def read_parquet_dir(directory: str | Path, pattern: str = "*.parquet") -> pd.DataFrame:
    """Concatenates every parquet part file under a directory, recursively.

    Args:
        directory: Directory to search. It may not exist.
        pattern: Glob pattern for the part files.

    Returns:
        All non-empty parts concatenated in sorted path order, or an empty frame
        if there are none.
    """
    parts = sorted(Path(directory).rglob(pattern)) if Path(directory).exists() else []
    frames = [pd.read_parquet(p) for p in parts]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def setup_logging(log_dir: str | Path, name: str, level: str = "INFO") -> None:
    """Sends Loguru output to stderr and to `<log_dir>/<name>_<timestamp>.log`.

    Removes any existing Loguru sinks first.

    Args:
        log_dir: Directory for the log file. It is created if missing.
        name: Prefix of the log file name.
        level: Minimum level for both sinks.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "{time:YYYY-MM-DD HH:mm:ss,SSS} {level} {name}: {message}"
    logger.remove()
    logger.add(sys.stderr, level=level, format=fmt)
    logger.add(
        log_dir / f"{name}_{time.strftime('%Y%m%dT%H%M%S')}.log",
        level=level,
        format=fmt,
    )
