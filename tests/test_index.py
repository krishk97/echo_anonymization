"""Tests for Script 1 (create_anonymization_plan.py) on synthetic DICOMs."""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom
import pytest

import create_anonymization_plan as script1
import synthetic
from src.anon_crosswalk import ANON_CROSSWALK_COLUMNS, AnonCrosswalk
from src.dicom_utils import (
    Region,
    cap_age,
    select_image_regions,
    select_video_regions,
    shift_date,
)
from src.io_utils import read_parquet_dir

DicomTree = tuple[Path, dict[str, str]]
MakeCfg = Callable[..., dict[str, Any]]


def run(cfg: dict[str, Any]) -> None:
    """Runs Script 1 with `cfg`."""
    script1.main(cfg)


def outputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads Script 1's outputs.

    Args:
        cfg: The config Script 1 ran with.

    Returns:
        (manifest, status, crosswalk read as strings).
    """
    secure = Path(cfg["paths"]["secure_dir"])
    return (
        read_parquet_dir(secure / "manifest"),
        read_parquet_dir(secure / "status"),
        pd.read_csv(
            secure / "anonymization_crosswalk.csv", dtype=str, keep_default_na=False
        ),
    )


def by_source(manifest: pd.DataFrame, paths: dict[str, str], name: str) -> pd.DataFrame:
    """Returns the manifest records of synthetic file `name`, sorted by RecordID."""
    return manifest[manifest["SourcePath"] == paths[name]].sort_values("RecordID")


# ---------------------------------------------------------------- filtering & status


def test_sop_filter_and_statuses(dicom_tree: DicomTree, make_cfg: MakeCfg) -> None:
    """Each excluded file gets its status; only included files reach the manifest.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    _, paths = dicom_tree
    cfg = make_cfg()
    run(cfg)
    manifest, status, _ = outputs(cfg)
    st = status.set_index("SourcePath")
    assert st.loc[paths["x_ct"], "Status"] == "excluded_sop_class"
    assert st.loc[paths["x_ct"], "Detail"] == "CT Image Storage"
    assert st.loc[paths["x_nopixels"], "Status"] == "excluded_no_pixel_data"
    assert st.loc[paths["x_no_patient_id"], "Status"] == "excluded_missing_patient_id"
    assert st.loc[paths["x_no_study_uid"], "Status"] == "excluded_missing_study_uid"
    assert st.loc[paths["b1_doppler_only"], "Status"] == "excluded_no_video_region"
    for name in ("x_img_noregions", "x_img_unusable_regions"):
        assert st.loc[paths[name], "Status"] == "excluded_no_image_region"
    assert st.loc[paths["x_corrupt"], "Status"] in {"error", "excluded_sop_class"}
    excluded = {paths[k] for k in paths if k.startswith("x_")}
    excluded |= {paths["b1_doppler_only"]}
    assert set(manifest["SourcePath"]) == set(paths.values()) - excluded
    assert set(manifest["SOPClass"]) <= set(cfg["index"]["sop_classes"])
    # notes.txt never discovered; every .dcm logged once
    assert len(status) == len(paths)


def test_max_num_frames_excludes_long_files(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """Files with more than `index.max_num_frames` frames are excluded.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    _, paths = dicom_tree
    cfg = make_cfg(max_num_frames=10)
    run(cfg)
    manifest, status, _ = outputs(cfg)
    st = status.set_index("SourcePath")
    for name in ("a1_s1_vid", "a1_s1_multi"):  # 12 frames
        assert st.loc[paths[name], "Status"] == "excluded_too_many_frames"
        assert paths[name] not in set(manifest["SourcePath"])
    assert st.loc[paths["a1_s2_vid"], "Status"] == "included"  # exactly 10 frames
    assert manifest["NumFrames"].max() == 10


# ---------------------------------------------------------------- regions


def test_one_record_per_region_and_image_bboxes(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """Videos get one record per kept region; images get 2D/1D region columns.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    _, paths = dicom_tree
    cfg = make_cfg()
    run(cfg)
    manifest, _, _ = outputs(cfg)

    multi = by_source(manifest, paths, "a1_s1_multi")
    # Color box (2) inside LEFT and ECG (3) dropped.
    assert list(multi["RegionIndex"]) == [0, 1]
    assert [i.rsplit("/", 1)[1] for i in multi["RecordID"]] == ["3_r0", "3_r1"]
    left = multi.iloc[0]
    # Max is inclusive in DICOM.
    assert (left.BBoxX0, left.BBoxY0, left.BBoxX1, left.BBoxY1) == (5, 10, 75, 100)
    assert (multi["ItemType"] == "video").all() and (multi["FPS"] == 25.0).all()

    img = by_source(manifest, paths, "a1_s1_img").iloc[0]
    assert img.ItemType == "image" and img.OutputPath.endswith("/1.png")
    assert (img.BBox2DX0, img.BBox2DY0, img.BBox2DX1, img.BBox2DY1) == (
        20,
        10,
        110,
        100,
    )
    assert (img.BBox1DX0, img.BBox1DY0, img.BBox1DX1, img.BBox1DY1) == (
        5,
        102,
        155,
        118,
    )

    noreg_vid = by_source(manifest, paths, "b1_vid_noregions").iloc[0]
    assert noreg_vid.RegionSource == "full_frame"
    noreg_box = (noreg_vid.BBoxX0, noreg_vid.BBoxY0, noreg_vid.BBoxX1, noreg_vid.BBoxY1)
    assert noreg_box == (0, 0, 160, 120)
    assert (img.RegionIndex2D, img.SpatialFormat2D, img.DataType2D) == (0, 1, 1)
    assert (img.RegionIndex1D, img.SpatialFormat1D, img.DataType1D) == (1, 3, 3)
    a2_img = by_source(manifest, paths, "a2_img").iloc[0]
    assert (a2_img.BBox2DX1, a2_img.BBox2DY1) == (110, 100)
    assert pd.isna(a2_img.BBox1DX0) and pd.isna(a2_img.RegionIndex1D)

    # Each manifest holds only its own type's region columns.
    secure = Path(cfg["paths"]["secure_dir"])
    videos = read_parquet_dir(secure / "manifest" / "video")
    images = read_parquet_dir(secure / "manifest" / "image")
    assert (videos["ItemType"] == "video").all()
    assert (images["ItemType"] == "image").all()
    assert not {"BBox2DX0", "BBox1DX0", "PhysicalDeltaX2D"} & set(videos.columns)
    assert not {"BBoxX0", "RegionIndex", "FPS", "PhysicalDeltaX"} & set(images.columns)
    # Calibration tags are flat scalar columns per region; no list/dict columns.
    assert (img.PhysicalDeltaX2D, img.PhysicalUnitsYDirection1D) == (0.05, 3)
    assert (left.PhysicalDeltaY, left.PhysicalUnitsXDirection) == (0.05, 3)
    assert pd.isna(noreg_vid.PhysicalDeltaX)  # full-frame fallback: no calibration
    for t in (videos, images):
        assert not t.map(lambda v: isinstance(v, list | dict)).any().any()


def test_region_selection_units() -> None:
    """Video region dedup and image 2D/1D selection on hand-made regions."""
    big = Region(0, 0, 0, 100, 100, 1, 1)
    inner_color = Region(1, 10, 10, 50, 50, 1, 2)
    other = Region(2, 100, 0, 200, 100, 1, 1)
    dup = Region(3, 100, 0, 200, 100, 1, 1)
    spectral = Region(4, 0, 100, 200, 150, 3, 3)
    gray_bar = Region(5, 0, 0, 300, 300, 1, 13)
    selected = select_video_regions([big, inner_color, other, dup, spectral])
    assert [r.index for r in selected] == [0, 2]
    two_d, one_d = select_image_regions([inner_color, big, spectral])
    assert two_d.index == 0 and one_d.index == 4
    assert select_image_regions([spectral]) == (None, spectral)
    # A 2D region that isn't tissue/color/unknown never becomes the 2D box.
    assert select_image_regions([gray_bar]) == (None, None)
    assert select_image_regions([gray_bar, big])[0] is big


# ---------------------------------------------------------------- naming


def test_output_path_collisions(dicom_tree: DicomTree, make_cfg: MakeCfg) -> None:
    """Clashing InstanceNumbers and missing ones get the documented suffixes.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    _, paths = dicom_tree
    cfg = make_cfg()
    run(cfg)
    manifest, _, _ = outputs(cfg)

    def name(k: str) -> str:
        return by_source(manifest, paths, k)["RecordID"].iloc[0].rsplit("/", 1)[1]

    assert name("a1_s1_vid") == "2"  # first claimer (series 1 sorts first)
    assert name("a1_s2_vid") == "2_s2"  # same InstanceNumber in series 2
    assert name("a1_s2_noinst") == "na0"  # missing InstanceNumber
    assert manifest["RecordID"].is_unique and manifest["OutputPath"].is_unique
    extensions = manifest["ItemType"].map({"video": ".avi", "image": ".png"})
    assert (manifest["OutputPath"] == manifest["RecordID"] + extensions).all()


def test_naming_is_deterministic(dicom_tree: DicomTree, make_cfg: MakeCfg) -> None:
    """Names don't depend on the run or on the chunk size.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    cfg_a, cfg_b, cfg_c = (
        make_cfg("a", chunk_size=4),
        make_cfg("b", chunk_size=4),
        make_cfg("c", chunk_size=100),
    )
    for cfg in (cfg_a, cfg_b, cfg_c):
        run(cfg)
    ids = [
        outputs(c)[0].set_index("SourcePath")["RecordID"].sort_index()
        for c in (cfg_a, cfg_b, cfg_c)
    ]
    pd.testing.assert_series_equal(ids[0], ids[1])
    # file names inside the study don't depend on the chunk size
    pd.testing.assert_series_equal(
        ids[0].str.rsplit("/", n=1).str[1], ids[2].str.rsplit("/", n=1).str[1]
    )


def test_collision_across_runs(
    tmp_path: Path, dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """A clashing file in a later run gets a suffix; earlier names never change.

    Args:
        tmp_path: pytest's per-test temporary directory.
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    root, paths = dicom_tree
    cfg = make_cfg()
    run(cfg)

    def pairs(m: pd.DataFrame) -> set[tuple[str, str]]:
        return set(zip(m["SourcePath"], m["RecordID"], strict=True))

    before = pairs(outputs(cfg)[0])
    study_a1 = pydicom.dcmread(
        paths["a1_s1_vid"], stop_before_pixels=True
    ).StudyInstanceUID
    late = synthetic.write(
        synthetic.make_dataset(
            patient_id="SYN-A",
            study_uid=study_a1,
            series_number=2,
            instance_number=2,
            n_frames=6,
        ),
        root / "zz_late" / "x.dcm",
    )
    run(cfg)
    after = pairs(outputs(cfg)[0])
    assert before < after
    assert [i for s, i in after - before] == [
        i for s, i in after if s == str(late.resolve())
    ]
    assert next(iter(after - before))[1].endswith("/2_s2-2")


# ---------------------------------------------------------------- crosswalk


def test_crosswalk_rows_and_patient_reuse(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """One crosswalk row per study; a patient keeps one ID and one date offset.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    _, paths = dicom_tree
    cfg = make_cfg()
    run(cfg)
    manifest, _, xw = outputs(cfg)
    assert list(xw.columns) == ANON_CROSSWALK_COLUMNS
    assert xw["StudyInstanceUID"].is_unique and len(xw) == 3  # A1, A2, B1
    pat_a = xw[xw["PatientID"] == "SYN-A"]
    assert pat_a["AnonPatientID"].nunique() == 1
    assert pat_a["DateOffsetDays"].nunique() == 1
    assert sorted(xw["AnonStudyID"]) == ["S000000", "S000001", "S000002"]
    offset = int(pat_a["DateOffsetDays"].iloc[0])
    lo, hi = cfg["index"]["date_offset_days"]
    assert lo <= offset <= hi

    a2 = by_source(manifest, paths, "a2_img").iloc[0]
    assert a2.PatientAgeCapped == "90+"
    assert a2.StudyDateShifted == shift_date("20210301", offset)
    assert by_source(manifest, paths, "a1_s1_img").iloc[0].PatientAgeCapped == "045Y"


def test_crosswalk_continues_after_restart(
    tmp_path: Path, dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """A later run appends new rows and reuses existing patient IDs and offsets.

    Args:
        tmp_path: pytest's per-test temporary directory.
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    root, _ = dicom_tree
    cfg = make_cfg()
    run(cfg)
    xw_path = Path(cfg["paths"]["secure_dir"]) / "anonymization_crosswalk.csv"
    first_bytes = xw_path.read_bytes()
    xw1 = pd.read_csv(xw_path, dtype=str, keep_default_na=False)

    # New run: a new study for an existing patient (in a later chunk) and a brand-new
    # patient.
    synthetic.write(
        synthetic.make_dataset(patient_id="SYN-A", study_uid="1.2.3.900", n_frames=6),
        root / "zz" / "a3.dcm",
    )
    synthetic.write(
        synthetic.make_dataset(patient_id="SYN-D", study_uid="1.2.3.901", n_frames=6),
        root / "zz" / "d1.dcm",
    )
    run(cfg)
    # Append-only, existing rows untouched.
    assert xw_path.read_bytes().startswith(first_bytes)
    xw2 = pd.read_csv(xw_path, dtype=str, keep_default_na=False)
    new = xw2.iloc[len(xw1) :].set_index("StudyInstanceUID")
    old_a = xw1[xw1["PatientID"] == "SYN-A"].iloc[0]
    assert new.loc["1.2.3.900", "AnonPatientID"] == old_a["AnonPatientID"]
    assert new.loc["1.2.3.900", "DateOffsetDays"] == old_a["DateOffsetDays"]
    assert new.loc["1.2.3.901", "AnonPatientID"] == "P000002"
    assert sorted(new["AnonStudyID"]) == ["S000003", "S000004"]

    run(cfg)  # nothing pending -> nothing changes
    assert len(pd.read_csv(xw_path, dtype=str)) == len(xw2)


def test_truncated_crosswalk_line_recovered(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """A partial final line from a crash is dropped and counters resume correctly.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    root, _ = dicom_tree
    cfg = make_cfg()
    run(cfg)
    xw_path = Path(cfg["paths"]["secure_dir"]) / "anonymization_crosswalk.csv"
    good = xw_path.read_bytes()
    with open(xw_path, "ab") as f:
        f.write(b'"SYN-Z","SYNTH^PART')  # crash mid-append
    xw = AnonCrosswalk(xw_path, (1, 2))
    assert xw_path.read_bytes() == good
    assert (xw.next_patient, xw.next_study) == (2, 3)

    synthetic.write(
        synthetic.make_dataset(patient_id="SYN-Z", study_uid="1.2.3.950", n_frames=6),
        root / "zz" / "z.dcm",
    )
    run(cfg)
    with open(xw_path, newline="") as f:
        rows = list(csv.reader(f))
    assert all(len(r) == len(ANON_CROSSWALK_COLUMNS) for r in rows)
    assert rows[-1][ANON_CROSSWALK_COLUMNS.index("AnonStudyID")] == "S000003"


# ---------------------------------------------------------------- crash / restart


class Boom(Exception):
    """Simulated crash."""


@pytest.mark.parametrize("crash_on", ["video", "image", "status"])
def test_restart_after_crash(
    dicom_tree: DicomTree,
    make_cfg: MakeCfg,
    monkeypatch: pytest.MonkeyPatch,
    crash_on: str,
) -> None:
    """A crash mid-chunk followed by a rerun gives the same outputs as a clean run.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
        monkeypatch: pytest's monkeypatch fixture.
        crash_on: Which part write of the 2nd chunk crashes.
    """
    clean = make_cfg("clean")
    run(clean)
    clean_manifest, clean_status, clean_xw = outputs(clean)

    cfg = make_cfg("crashy")
    real_write = script1.atomic_write_parquet
    calls = {"n": 0}

    def flaky_write(df: pd.DataFrame, path: str | Path) -> None:
        # crash on the 2nd chunk, after its crosswalk append, at the chosen write
        if (
            Path(path).parent.name == crash_on
            and Path(path).name == "chunk-000001.parquet"
        ):
            raise Boom()
        calls["n"] += 1
        return real_write(df, path)

    monkeypatch.setattr(script1, "atomic_write_parquet", flaky_write)
    with pytest.raises(Boom):
        run(cfg)
    monkeypatch.setattr(script1, "atomic_write_parquet", real_write)
    run(cfg)

    manifest, status, xw = outputs(cfg)
    assert manifest["RecordID"].is_unique and status["SourcePath"].is_unique
    assert xw["StudyInstanceUID"].is_unique
    assert len(manifest) == len(clean_manifest) and len(status) == len(clean_status)
    pd.testing.assert_series_equal(
        manifest.set_index("SourcePath")["RecordID"].sort_index(),
        clean_manifest.set_index("SourcePath")["RecordID"].sort_index(),
    )
    secure = Path(cfg["paths"]["secure_dir"])
    n_status = len(list((secure / "status").glob("*.parquet")))
    for item_type in ("video", "image"):
        parts = (secure / "manifest" / item_type).glob("*.parquet")
        assert len(list(parts)) == n_status


def test_multiprocessing_matches_serial(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """A worker pool gives the same RecordIDs as a serial run.

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    serial, parallel = make_cfg("serial"), make_cfg("parallel", num_workers=2)
    run(serial)
    run(parallel)
    a, b = outputs(serial)[0], outputs(parallel)[0]
    pd.testing.assert_series_equal(
        a.set_index("SourcePath")["RecordID"].sort_index(),
        b.set_index("SourcePath")["RecordID"].sort_index(),
    )


# ---------------------------------------------------------------- PHI


def test_no_phi_or_real_keys_in_manifest(
    dicom_tree: DicomTree, make_cfg: MakeCfg
) -> None:
    """No identifying columns or fake PHI reach the manifest (besides SourcePath).

    Args:
        dicom_tree: The synthetic tree fixture.
        make_cfg: The config factory fixture.
    """
    cfg = make_cfg()
    run(cfg)
    manifest, status, _ = outputs(cfg)
    forbidden_cols = {
        "PatientID",
        "PatientName",
        "PatientBirthDate",
        "AccessionNumber",
        "StudyInstanceUID",
        "StudyID",
        "StudyDate",
        "PatientAge",
        "InstitutionName",
        "StudyDescription",
    }
    assert not forbidden_cols & set(manifest.columns)
    text = manifest.drop(columns=["SourcePath"]).astype(str).to_string()
    for phi in synthetic.FAKE_PHI + ["SYN-A", "SYN-B", "ACCSYN"]:
        assert phi not in text
    assert set(status.columns) == {"SourcePath", "Status", "Detail", "NumRecords"}


def test_age_and_date_helpers() -> None:
    """`cap_age` and `shift_date` on valid and invalid values."""
    assert cap_age("089Y", 90) == "089Y"
    assert cap_age("090Y", 90) == "90+"
    assert cap_age("006M", 90) == "006M"
    assert cap_age("garbage", 90) is None and cap_age("", 90) is None
    assert shift_date("20200101", 1) == "20200102"
    assert shift_date("20200101120000.5", 31) == "20200201120000.5"
    assert shift_date("bad", 5) is None


@pytest.mark.parametrize("num_threads", [1, 8])
def test_find_files_matches_rglob(tmp_path: Path, num_threads: int) -> None:
    """The threaded scandir walk finds the same files as the old `rglob` version.

    Args:
        tmp_path: pytest's temporary directory.
        num_threads: Thread count passed to `find_files_by_extension`.
    """
    root, outside = tmp_path / "root", tmp_path / "outside"
    for d in (root / "a" / "b", root / "c", outside):
        d.mkdir(parents=True)
    for f in ("x.dcm", "a/y.DCM", "a/b/z.dcm", "a/b/n.txt", "c/.dcm", "c/noext"):
        (root / f).touch()
    (outside / "o.dcm").touch()
    (root / "linked_dir").symlink_to(outside, target_is_directory=True)
    (root / "c" / "linked.dcm").symlink_to(outside / "o.dcm")

    expected = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".dcm"
    )
    found = sorted(script1.find_files_by_extension(root, "dcm", num_threads))
    assert found == expected
    assert root / "c" / "linked.dcm" in found
    assert len(found) == 4
