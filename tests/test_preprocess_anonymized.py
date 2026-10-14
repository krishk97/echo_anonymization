"""Tests for run_preprocessing_on_anonymized.py on a MIMIC-IV-ECHO-style dataset."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import run_echo_preprocessing_and_anonymization_with_classifier as script2
import run_preprocessing_on_anonymized as anon
import synthetic
from test_preprocess import stub_models

MakeCfg = Callable[..., dict[str, Any]]
# Synthetic patient/study folders -> MIMIC-style integer subject_id / study_id
SUBJECTS = {"patA": "10000001", "patB": "10000002"}
STUDIES = {"studyA1": "90000001", "studyA2": "90000002", "studyB1": "90000003"}


def mimic_dataset(root: Path) -> pd.DataFrame:
    """Writes the synthetic tree under `root/files` and returns its record list.

    Only the patient A/B files are listed, with MIMIC-IV-ECHO column names and
    root-relative paths with a leading slash (`/files/patA/studyA1/s1_i1.dcm`).

    Args:
        root: Dataset root (where the record list would live).

    Returns:
        The record list (`dicom_filepath`, `subject_id`, `study_id`).
    """
    synthetic.build_tree(root / "files")
    rows = []
    for path in sorted((root / "files").glob("pat*/study*/*.dcm")):
        patient, study = path.parent.parent.name, path.parent.name
        rows.append(
            {
                "dicom_filepath": "/" + str(path.relative_to(root)),
                "subject_id": int(SUBJECTS[patient]),
                "study_id": int(STUDIES[study]),
            }
        )
    return pd.DataFrame(rows)


def small_cfg(make_cfg: MakeCfg) -> dict[str, Any]:
    """Returns a config for one in-process-friendly CPU process, no workers."""
    cfg = make_cfg()
    cfg["preprocess"].update(num_workers=0, num_cpu_procs=1, num_gpus=1)
    return cfg


def test_mimic_style_csv_lists(tmp_path: Path, make_cfg: MakeCfg) -> None:
    """CSV lists with MIMIC columns give CSV outputs with the list's IDs.

    Args:
        tmp_path: pytest's per-test temporary directory.
        make_cfg: The config factory fixture.
    """
    root = tmp_path / "mimic"
    mimic_dataset(root).to_csv(root / "echo-record-list.csv", index=False)
    pd.DataFrame(
        {
            "study_id": [*STUDIES.values(), "99999999"],
            "measurement_id": ["m1", "", "m3", "m4"],
        }
    ).to_csv(root / "echo-study-list.csv", index=False)
    out, work = tmp_path / "out", tmp_path / "work"

    assert anon.main(
        small_cfg(make_cfg),
        root / "echo-record-list.csv",
        out,
        work,
        study_list=root / "echo-study-list.csv",
        path_column="dicom_filepath",
        patient_column="subject_id",
        study_column="study_id",
    )

    assert not list(out.glob("*.parquet"))
    videos = pd.read_csv(out / "echo-video-list.csv", dtype=str)
    images = pd.read_csv(out / "echo-image-list.csv", dtype=str)
    both = pd.concat([videos, images])
    assert len(videos) and len(images)
    assert set(both["PatientID"]) <= set(SUBJECTS.values())
    assert set(both["StudyID"]) <= set(STUDIES.values())
    assert both["SourceFilePath"].str.startswith("/files/pat").all()
    assert (both["Status"] == "preprocessed").all()
    assert "SourcePath" not in both.columns
    for path in both["FilePath"]:
        assert (out / "files" / path).exists(), path
    # Already-shifted dates are copied unchanged (offset 0).
    a2 = images[images["StudyID"] == STUDIES["studyA2"]]
    assert (a2["StudyDateShifted"] == "20210301").all()

    studies = pd.read_csv(out / "echo-study-list.csv", dtype=str, keep_default_na=False)
    assert list(studies.columns) == ["study_id", "measurement_id"]
    assert set(studies["study_id"]) == set(both["StudyID"])
    assert "99999999" not in set(studies["study_id"])


def test_parquet_lists_with_classify(
    tmp_path: Path, make_cfg: MakeCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parquet list gives parquet outputs; --classify adds the view columns.

    The GPU `run` is replaced by an in-process run with stub models. A second
    call reuses the work manifest.

    Args:
        tmp_path: pytest's per-test temporary directory.
        make_cfg: The config factory fixture.
        monkeypatch: pytest's monkeypatch fixture.
    """

    def fake_run(cfg: dict[str, Any], retry_errors: bool, log_level: str) -> bool:
        pending = script2.pending_records(cfg, retry_errors)
        for rank, files in enumerate(script2.shard_files(pending, 1)):
            script2.run_rank(rank, files, cfg, stub_models())
        return True

    monkeypatch.setattr(script2, "run", fake_run)
    root = tmp_path / "data"
    records = mimic_dataset(root).rename(
        columns={
            "dicom_filepath": "FilePath",
            "subject_id": "PatientID",
            "study_id": "StudyID",
        }
    )
    records.to_parquet(root / "list.parquet", index=False)
    out, work = tmp_path / "out", tmp_path / "work"
    cfg = small_cfg(make_cfg)

    assert anon.main(cfg, root / "list.parquet", out, work, classify=True)
    videos = pd.read_parquet(out / "echo-video-list.parquet")
    assert sorted(p.name for p in out.glob("echo-*")) == [
        "echo-image-list.parquet",
        "echo-study-list.csv",
        "echo-video-list.parquet",
    ]
    assert (videos["Status"] == "success").all()
    assert "View_EchoFine" in videos.columns
    assert set(videos["PatientID"]) <= set(SUBJECTS.values())

    manifest_mtime = (work / "status" / anon.MANIFEST_PART).stat().st_mtime
    assert anon.main(cfg, root / "list.parquet", out, work, classify=True)
    assert (work / "status" / anon.MANIFEST_PART).stat().st_mtime == manifest_mtime


def test_missing_column_raises(tmp_path: Path, make_cfg: MakeCfg) -> None:
    """A record list without the path column is rejected before any work.

    Args:
        tmp_path: pytest's per-test temporary directory.
        make_cfg: The config factory fixture.
    """
    root = tmp_path / "mimic"
    mimic_dataset(root).to_csv(root / "echo-record-list.csv", index=False)
    with pytest.raises(ValueError, match="FilePath"):
        anon.main(
            small_cfg(make_cfg),
            root / "echo-record-list.csv",
            tmp_path / "out",
            tmp_path / "work",
        )
