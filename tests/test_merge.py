"""Tests for Script 2 `merge`: final video/image tables, study list, cleanup."""

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from loguru import logger

import create_anonymization_plan as script1
import run_echo_preprocessing_and_anonymization_with_classifier as script2
import synthetic
from src.dicom_utils import shift_date
from src.io_utils import read_parquet_dir
from test_preprocess import run_all, stub_models

DicomTree = tuple[Path, dict[str, str]]
MakeCfg = Callable[..., dict[str, Any]]


def setup(dicom_tree: DicomTree, make_cfg: MakeCfg) -> dict[str, Any]:
    """Runs Script 1, then Script 2 with one error that a retry fixes.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.

    Returns:
        The config both scripts ran with.
    """
    cfg = make_cfg()
    cfg["preprocess"].update(num_workers=0, flush_every=3, num_gpus=2)
    script1.main(cfg)
    run_all(cfg, stub_models(fail_item=[1]))
    run_all(cfg, stub_models(), retry_errors=True)
    return cfg


def test_merge_dedupes_and_keeps_only_allowed_columns(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """Each final table has one row per record and only its allowlisted columns.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    cfg = setup(dicom_tree, make_cfg)
    videos, images = script2.merge(cfg)
    out = Path(cfg["paths"]["output_root"])
    assert videos.equals(pd.read_parquet(out / "echo-video-list.parquet"))
    assert images.equals(pd.read_parquet(out / "echo-image-list.parquet"))
    assert len(videos) and len(images)
    table = pd.concat([videos, images], ignore_index=True)
    studies = pd.read_csv(out / "echo-study-list.csv")
    assert list(studies.columns) == ["PatientID", "StudyID", "StudyDateTime"]
    assert set(studies["StudyID"]) == set(table["StudyID"])

    manifest = read_parquet_dir(Path(cfg["paths"]["secure_dir"]) / "manifest")
    assert table["RecordID"].is_unique and len(table) == len(manifest)
    # The retried error was superseded by the later row.
    assert (table["Status"] == "success").all()

    icfg = cfg["index"]
    common = (
        set(script2.MANIFEST_KEEP)
        | set(script2.RESULT_KEEP)
        | set(icfg["metadata_allowlist"])
    )
    common |= {f"{k}Shifted" for k in icfg["date_fields"]}
    # Renamed at merge.
    common -= {"AnonPatientID", "AnonStudyID", "OutputPath"}
    common |= {"PatientID", "StudyID", "FilePath"}
    for t, type_keep in ((videos, script2.VIDEO_KEEP), (images, script2.IMAGE_KEEP)):
        non_model = {
            c for c in t.columns if not c.startswith(script2.MODEL_COL_PREFIXES)
        }
        assert not non_model - common - set(type_keep)
        assert not {"SourcePath", "OutputPath", "ItemType"} & set(t.columns)
    # Each table holds only its own type's region columns.
    assert not {"BBox2DX0", "BBox1DX0", "PhysicalDeltaX2D"} & set(videos.columns)
    assert not {"BBoxX0", "RegionIndex", "FPS", "PhysicalDeltaX"} & set(images.columns)
    assert {"RegionIndex2D", "PhysicalDeltaX2D", "PhysicalDeltaY1D"} <= set(
        images.columns
    )
    assert {"RegionIndex", "PhysicalDeltaX"} <= set(videos.columns)
    assert table.set_index("RecordID")["FilePath"].equals(
        manifest.set_index("RecordID")["OutputPath"].loc[table["RecordID"]]
    )
    assert table["PatientID"].str.fullmatch(r"P[0-9a-f]{16}").all()
    assert table["StudyID"].str.fullmatch(r"S[0-9a-f]{16}").all()

    text = table.astype(str).to_string()
    for phi in synthetic.FAKE_PHI + ["SYN-A", "SYN-B", "ACCSYN", str(dicom_tree[0])]:
        assert phi not in text


def test_merge_counts_missing_media(
    dicom_tree: DicomTree, make_cfg: MakeCfg, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing media file is counted in the QA log and keeps `results/`.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
        caplog: pytest's log capture fixture (fed through a Loguru sink).
    """
    cfg = setup(dicom_tree, make_cfg)
    media = Path(cfg["paths"]["output_root"]) / "files"
    next(media.rglob("*.avi")).unlink()
    sink = logger.add(caplog.handler, level="INFO", format="{message}")
    try:
        script2.merge(cfg)
    finally:
        logger.remove(sink)
    assert "missing media file: 1" in caplog.text
    # Incomplete merge keeps the results.
    assert (Path(cfg["paths"]["output_root"]) / "results").is_dir()


def crosswalk(cfg: dict[str, Any]) -> pd.DataFrame:
    """Reads the crosswalk as strings, indexed by AnonStudyID."""
    path = Path(cfg["paths"]["secure_dir"]) / "anonymization_crosswalk.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("AnonStudyID")


def test_study_datetime_shifted_per_patient(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """StudyDateTime is the real date and time shifted by the patient's offset.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    cfg = setup(dicom_tree, make_cfg)
    script2.merge(cfg)
    cw = crosswalk(cfg)
    assert (cw.groupby("PatientID")["DateOffsetDays"].nunique() == 1).all()
    studies = pd.read_csv(
        Path(cfg["paths"]["output_root"]) / "echo-study-list.csv",
        parse_dates=["StudyDateTime"],
    )
    assert len(studies) == 3
    for s in studies.itertuples():
        real = cw.loc[s.StudyID]
        expected = dt.datetime.strptime(real.StudyDate + real.StudyTime, "%Y%m%d%H%M%S")
        assert s.StudyDateTime == expected + dt.timedelta(days=int(real.DateOffsetDays))
        assert s.PatientID == real.AnonPatientID
    # Intervals between a patient's studies are preserved
    anon_a = cw.loc[cw["PatientID"] == "SYN-A", "AnonPatientID"].iloc[0]
    a = studies[studies["PatientID"] == anon_a]
    interval = a["StudyDateTime"].max() - a["StudyDateTime"].min()
    assert interval.days == (dt.date(2021, 3, 1) - dt.date(2020, 1, 15)).days


def test_acquisition_datetime_shifted_with_crosswalk_offset(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """AcquisitionDateTimeShifted uses the crosswalk offset and keeps the time.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    cfg = setup(dicom_tree, make_cfg)
    table = pd.concat(script2.merge(cfg))
    cw = crosswalk(cfg)
    for rec in table.itertuples():
        real = cw.loc[rec.StudyID]
        raw = real.StudyDate + "101730.250000"  # synthetic AcquisitionDateTime
        expected = shift_date(raw, int(real.DateOffsetDays))
        assert rec.AcquisitionDateTimeShifted == expected
        assert rec.AcquisitionDateTimeShifted[:8] == rec.StudyDateShifted
        assert rec.AcquisitionDateTimeShifted[8:] == "101730.250000"


def test_complete_merge_deletes_results(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """A complete merge deletes `results/` and writes both final files.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    cfg = setup(dicom_tree, make_cfg)
    out = Path(cfg["paths"]["output_root"])
    script2.merge(cfg)
    assert not (out / "results").exists()
    assert (out / "echo-video-list.parquet").exists()
    assert (out / "echo-image-list.parquet").exists()
    assert (out / "echo-study-list.csv").exists()


def test_merge_with_errors_keeps_results(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """A merge with error rows keeps `results/` so `run` can resume.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    cfg = make_cfg()
    cfg["preprocess"].update(num_workers=0, flush_every=3, num_gpus=2)
    script1.main(cfg)
    run_all(cfg, stub_models(fail_item=[1]))
    table = pd.concat(script2.merge(cfg))
    assert (table["Status"] == "error").sum() == 1
    assert (Path(cfg["paths"]["output_root"]) / "results").is_dir()


def test_pipeline_version_bump_requeues_and_keeps_results(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """After a version bump, old results are pending again and merge keeps them.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    cfg = setup(dicom_tree, make_cfg)
    n_records = len(read_parquet_dir(Path(cfg["paths"]["secure_dir"]) / "manifest"))
    assert script2.pending_records(cfg, retry_errors=False).empty

    cfg["preprocess"]["pipeline_version"] = "v2"
    assert len(script2.pending_records(cfg, retry_errors=False)) == n_records
    script2.merge(cfg)
    assert (Path(cfg["paths"]["output_root"]) / "results").is_dir()

    run_all(cfg, stub_models())
    assert script2.pending_records(cfg, retry_errors=False).empty
    table = pd.concat(script2.merge(cfg))
    assert (table["PipelineVersion"] == "v2").all()
    assert not (Path(cfg["paths"]["output_root"]) / "results").exists()
