import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.io_utils import load_config  # noqa: E402
import synthetic  # noqa: E402


@pytest.fixture
def dicom_tree(tmp_path):
    root = tmp_path / "dicoms"
    return root, synthetic.build_tree(root)


@pytest.fixture
def make_cfg(tmp_path):
    """Config from configs/config.yaml with paths redirected into tmp_path."""
    base = load_config(ROOT / "configs" / "config.yaml")

    def _make(name="run", dicom_root=None, **index_overrides):
        cfg = copy.deepcopy(base)
        cfg["paths"]["dicom_root"] = str(dicom_root or tmp_path / "dicoms")
        cfg["paths"]["secure_dir"] = str(tmp_path / name / "secure")
        cfg["paths"]["output_root"] = str(tmp_path / name / "output")
        cfg["index"].update({"num_workers": 1, "chunk_size": 4, **index_overrides})
        return cfg

    return _make
