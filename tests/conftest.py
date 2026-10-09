"""Shared pytest fixtures: a synthetic DICOM tree and configs under tmp_path."""

import copy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import synthetic  # noqa: E402
from src.io_utils import load_config  # noqa: E402


@pytest.fixture
def dicom_tree(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Writes the standard synthetic dataset under `tmp_path/dicoms`.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        (tree root, {name: path}) as from `synthetic.build_tree`.
    """
    root = tmp_path / "dicoms"
    return root, synthetic.build_tree(root)


@pytest.fixture
def make_cfg(tmp_path: Path) -> Callable[..., dict[str, Any]]:
    """Provides a factory for configs with paths redirected into `tmp_path`.

    The factory takes `name` (subdirectory for secure_dir/output_root),
    `dicom_root` (default `tmp_path/dicoms`) and keyword overrides for the `index`
    section, and returns a copy of configs/config.yaml.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The config factory.
    """
    base = load_config(ROOT / "configs" / "config.yaml")

    def _make(
        name: str = "run", dicom_root: str | Path | None = None, **index_overrides: Any
    ) -> dict[str, Any]:
        cfg = copy.deepcopy(base)
        cfg["paths"]["dicom_root"] = str(dicom_root or tmp_path / "dicoms")
        cfg["paths"]["secure_dir"] = str(tmp_path / name / "secure")
        cfg["paths"]["output_root"] = str(tmp_path / name / "output")
        cfg["index"].update({"num_workers": 1, "chunk_size": 4, **index_overrides})
        return cfg

    return _make
