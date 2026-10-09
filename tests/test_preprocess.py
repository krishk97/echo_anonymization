"""Script 2 tests on CPU with stub models (no GPU, no weights)."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import create_anonymization_plan as script1
import run_echo_preprocessing_and_anonymization_with_classifier as script2
from src.io_utils import read_parquet_dir

FINE_VIEWS = [f"fine{i}" for i in range(4)]
PRIME_VIEWS = [f"prime{i}" for i in range(3)]

Indexed = tuple[dict[str, Any], dict[str, str]]


def stub_models(
    fail_item: list[int] | None = None, calls: list[tuple[int, ...]] | None = None
) -> script2.Models:
    """Builds deterministic fake classifiers.

    EchoFine output depends on the input, so sharing can be checked.

    Args:
        fail_item: One-element counter; EchoPrime raises while it is positive,
            decrementing it each time.
        calls: If given, EchoPrime appends each input shape to it.

    Returns:
        The stub models.
    """

    def echofine(x: np.ndarray) -> np.ndarray:
        p = np.full((len(x), len(FINE_VIEWS)), 0.1)
        p[:, 0] = x.mean() / 255.0
        return p

    def echoprime(x: np.ndarray) -> np.ndarray:
        if calls is not None:
            calls.append(x.shape)
        if fail_item is not None and fail_item[0]:
            fail_item[0] -= 1
            raise RuntimeError("stub failure")
        return np.tile(np.array([0.2, 0.5, 0.3]), (len(x), 1))

    return script2.Models(echofine, echoprime, FINE_VIEWS, PRIME_VIEWS)


@pytest.fixture
def indexed(
    dicom_tree: tuple[Path, dict[str, str]], make_cfg: Callable[..., dict[str, Any]]
) -> Indexed:
    """Indexes the synthetic tree with Script 1.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.

    Returns:
        (cfg, {name: path}).
    """
    _, paths = dicom_tree
    cfg = make_cfg()
    cfg["preprocess"].update(num_workers=0, flush_every=2, num_gpus=2)
    script1.main(cfg)
    return cfg, paths


def run_all(
    cfg: dict[str, Any], models: script2.Models, retry_errors: bool = False
) -> None:
    """Runs every rank of Script 2 in this process, one after another."""
    pending = script2.pending_records(cfg, retry_errors)
    shards = script2.shard_files(pending, cfg["preprocess"]["num_gpus"])
    for rank, files in enumerate(shards):
        script2.run_rank(rank, files, cfg, models)


def results(cfg: dict[str, Any]) -> pd.DataFrame:
    """Reads all result parts."""
    return read_parquet_dir(Path(cfg["paths"]["output_root"]) / "results")


def manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    """Reads all manifest parts."""
    return read_parquet_dir(Path(cfg["paths"]["secure_dir"]) / "manifest")


def test_media_and_result_rows(indexed: Indexed) -> None:
    """Every record gets a success row with view columns and a saved media file.

    Args:
        indexed: The indexed tree fixture.
    """
    cfg, paths = indexed
    run_all(cfg, stub_models())
    res, man = results(cfg), manifest(cfg)
    assert sorted(res["RecordID"]) == sorted(man["RecordID"])
    assert (res["Status"] == "success").all(), res[["RecordID", "Status", "ErrorType"]]
    assert {
        "EchoFine_View=fine0",
        "View_EchoFine",
        "ViewScore_EchoFine",
        "EchoPrime_View=prime1",
        "View_EchoPrime",
        "ViewScore_EchoPrime",
        "PipelineVersion",
        "Timestamp",
    } <= set(res.columns)
    assert not any(c.endswith("_std") for c in res.columns)
    assert (res["View_EchoPrime"] == "prime1").all()
    assert (res["ViewScore_EchoPrime"] == res["EchoPrime_View=prime1"]).all()

    media = Path(cfg["paths"]["output_root"]) / "files"
    for rec in man.itertuples():
        path = media / rec.OutputPath
        assert path.exists(), rec.OutputPath
        if rec.ItemType == "video":
            cap = cv2.VideoCapture(str(path))
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == rec.NumFrames
            cap.release()
    assert not list(media.rglob("*.tmp.*"))


def test_image_masked_outside_regions(indexed: Indexed) -> None:
    """A saved image keeps the full frame but is zero outside its 2D/1D boxes.

    Args:
        indexed: The indexed tree fixture.
    """
    cfg, paths = indexed
    run_all(cfg, stub_models())
    rec = manifest(cfg).query("SourcePath == @paths['a1_s1_img']").iloc[0]
    img = np.asarray(
        Image.open(Path(cfg["paths"]["output_root"]) / "files" / rec.OutputPath)
    )
    assert img.shape == (120, 160, 3)  # full frame kept
    keep = np.zeros(img.shape[:2], bool)
    for p in ("BBox2D", "BBox1D"):
        keep[rec[f"{p}Y0"] : rec[f"{p}Y1"], rec[f"{p}X0"] : rec[f"{p}X1"]] = True
    # Burned-in header bar (rows 0-5) and the ECG strip are gone.
    assert img[~keep].max() == 0
    assert img[keep].max() > 0


def test_multi_region_shares_echofine(indexed: Indexed) -> None:
    """All region records of one file share the file's EchoFine output.

    Args:
        indexed: The indexed tree fixture.
    """
    cfg, paths = indexed
    run_all(cfg, stub_models())
    man = manifest(cfg).query("SourcePath == @paths['a1_s1_multi']")
    res = results(cfg).set_index("RecordID").loc[man["RecordID"]]
    assert len(res) == 2 and res["EchoFine_View=fine0"].nunique() == 1


def test_oversize_skipped_without_decoding(
    indexed: Indexed, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversize file is marked skipped and never decoded.

    Args:
        indexed: The indexed tree fixture.
        monkeypatch: pytest's monkeypatch fixture.
    """
    cfg, paths = indexed
    man = manifest(cfg)
    big = man["SourcePath"] == paths["a1_s1_vid"]
    pending = man.assign(Oversize=big)
    decoded = []
    real_decode = script2.decode_frames
    monkeypatch.setattr(
        script2, "decode_frames", lambda p: decoded.append(p) or real_decode(p)
    )
    script2.run_rank(0, script2.shard_files(pending, 1)[0], cfg, stub_models())
    res = results(cfg).set_index("RecordID")
    assert res.loc[man.loc[big, "RecordID"].iloc[0], "Status"] == "skipped_oversize"
    assert paths["a1_s1_vid"] not in decoded


def test_undecodable_file_is_skipped(indexed: Indexed) -> None:
    """A file corrupted after indexing gets `skipped_undecodable` with an error type.

    Args:
        indexed: The indexed tree fixture.
    """
    cfg, paths = indexed
    Path(paths["a1_s1_vid"]).write_bytes(b"\x00" * 50)  # corrupted after indexing
    run_all(cfg, stub_models())
    man = manifest(cfg)
    rec_id = man.query("SourcePath == @paths['a1_s1_vid']")["RecordID"].iloc[0]
    row = results(cfg).set_index("RecordID").loc[rec_id]
    assert row["Status"] == "skipped_undecodable" and row["ErrorType"]


def test_shard_files_by_source_and_size(indexed: Indexed) -> None:
    """Sharding keeps a file's records together and deals the largest file first.

    Args:
        indexed: The indexed tree fixture.
    """
    cfg, _ = indexed
    man = manifest(cfg)
    shards = script2.shard_files(man, 2)
    sources = [src for shard in shards for src, _ in shard]
    # Each file exactly once.
    assert sorted(sources) == sorted(man["SourcePath"].unique())
    assert abs(len(shards[0]) - len(shards[1])) <= 1
    sizes = man.groupby("SourcePath")["DecodedBytes"].first()
    assert sizes[shards[0][0][0]] == sizes.max()  # largest file dealt first
    multi = [recs for shard in shards for _, recs in shard if len(recs) == 2]
    assert len(multi) == 1  # both regions of the multi-region video travel together


class Crash(BaseException):
    """Escapes the per-record `except Exception`, like a killed process."""


def test_restart_after_crash(indexed: Indexed, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a crash, a rerun finishes the remaining records without duplicates.

    Args:
        indexed: The indexed tree fixture.
        monkeypatch: pytest's monkeypatch fixture.
    """
    cfg, _ = indexed
    cfg["preprocess"]["flush_every"] = 1
    real_save, saved = script2.save_media, []

    def crashing_save(*args: Any, **kwargs: Any) -> None:
        if len(saved) == 3:
            raise Crash()
        saved.append(args[0])
        return real_save(*args, **kwargs)

    monkeypatch.setattr(script2, "save_media", crashing_save)
    with pytest.raises(Crash):
        run_all(cfg, stub_models())
    assert len(results(cfg)) == 3  # rows only for fully finished items

    monkeypatch.setattr(script2, "save_media", real_save)
    run_all(cfg, stub_models())
    res = results(cfg)
    assert res["RecordID"].is_unique
    assert sorted(res["RecordID"]) == sorted(manifest(cfg)["RecordID"])

    run_all(cfg, stub_models())  # nothing pending
    assert len(results(cfg)) == len(res)


def test_retry_errors(indexed: Indexed) -> None:
    """Error rows are redone only with `retry_errors`.

    Args:
        indexed: The indexed tree fixture.
    """
    cfg, _ = indexed
    run_all(cfg, stub_models(fail_item=[1]))
    first = script2.latest_results(results(cfg))
    assert (first["Status"] == "error").sum() == 1

    calls = []
    run_all(cfg, stub_models(calls=calls))  # without the flag, errors are done
    assert calls == []
    run_all(cfg, stub_models(calls=calls), retry_errors=True)
    assert len(calls) == 1
    latest = script2.latest_results(results(cfg))
    assert (latest["Status"] == "success").all() and len(latest) == len(first)


def test_summarize_score() -> None:
    """`summarize` scores each view as mean / (1 + std) and picks the best."""
    probits = np.array([[0.2, 0.8], [0.6, 0.4]])
    out = script2.summarize("EchoFine", probits, ["a", "b"])
    assert out["EchoFine_View=a"] == pytest.approx(0.4 / 1.2)
    assert out["EchoFine_View=b"] == pytest.approx(0.6 / 1.2)
    assert out["View_EchoFine"] == "b"
    assert out["ViewScore_EchoFine"] == out["EchoFine_View=b"]


class PydicomStateProbe(Dataset):
    """One-item dataset that reports whether pydicom's logger is disabled."""

    def __len__(self) -> int:
        """Returns 1."""
        return 1

    def __getitem__(self, index: int) -> bool:
        """Returns `logging.getLogger("pydicom").disabled` in the worker."""
        return logging.getLogger("pydicom").disabled


def test_dataloader_workers_silence_pydicom() -> None:
    """Spawned DataLoader workers run `quiet_pydicom()` via `_init_worker`."""
    loader = DataLoader(
        PydicomStateProbe(),
        batch_size=None,
        num_workers=1,
        multiprocessing_context="spawn",
        worker_init_fn=script2._init_worker,
    )
    assert list(loader) == [True]


def test_run_rank_with_spawned_workers(indexed: Indexed) -> None:
    """`run_rank` works with real DataLoader worker processes.

    Args:
        indexed: The indexed tree fixture.
    """
    cfg, _ = indexed
    cfg["preprocess"].update(num_workers=1, num_gpus=1)
    run_all(cfg, stub_models())
    res = results(cfg)
    assert sorted(res["RecordID"]) == sorted(manifest(cfg)["RecordID"])
    assert (res["Status"] == "success").all()
