import csv
from pathlib import Path

import pandas as pd
import pydicom
import pytest

import create_anonymization_plan as script1
import synthetic
from src.anon_crosswalk import ANON_CROSSWALK_COLUMNS, AnonCrosswalk
from src.io_utils import read_parquet_dir
from src.dicom_utils import Region, cap_age, select_image_regions, select_video_regions, shift_date


def run(cfg):
    return script1.main(cfg)


def outputs(cfg):
    secure = Path(cfg["paths"]["secure_dir"])
    return (read_parquet_dir(secure / "manifest"), read_parquet_dir(secure / "status"),
            pd.read_csv(secure / "anonymization_crosswalk.csv", dtype=str, keep_default_na=False))


def by_source(manifest, paths, name):
    return manifest[manifest["SourcePath"] == paths[name]].sort_values("RecordID")


# ---------------------------------------------------------------- filtering & status

def test_sop_filter_and_statuses(dicom_tree, make_cfg):
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
    assert st.loc[paths["x_corrupt"], "Status"] in {"error", "excluded_sop_class"}
    excluded = {paths[k] for k in paths if k.startswith("x_")} | {paths["b1_doppler_only"]}
    assert set(manifest["SourcePath"]) == set(paths.values()) - excluded
    assert set(manifest["SOPClass"]) <= set(cfg["index"]["sop_classes"])
    assert len(status) == len(paths)  # notes.txt never discovered; every .dcm logged once


def test_max_num_frames_excludes_long_files(dicom_tree, make_cfg):
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

def test_one_record_per_region_and_image_bboxes(dicom_tree, make_cfg):
    _, paths = dicom_tree
    cfg = make_cfg()
    run(cfg)
    manifest, _, _ = outputs(cfg)

    multi = by_source(manifest, paths, "a1_s1_multi")
    assert list(multi["RegionIndex"]) == [0, 1]  # color box (2) inside LEFT and ECG (3) dropped
    assert [i.rsplit("/", 1)[1] for i in multi["RecordID"]] == ["3_r0", "3_r1"]
    left = multi.iloc[0]
    assert (left.BBoxX0, left.BBoxY0, left.BBoxX1, left.BBoxY1) == (5, 10, 75, 100)  # max is inclusive in DICOM
    assert (multi["ItemType"] == "video").all() and (multi["FPS"] == 25.0).all()

    img = by_source(manifest, paths, "a1_s1_img").iloc[0]
    assert img.ItemType == "image" and img.OutputPath.endswith("/1.png")
    assert (img.BBox2DX0, img.BBox2DY0, img.BBox2DX1, img.BBox2DY1) == (20, 10, 110, 100)
    assert (img.BBox1DX0, img.BBox1DY0, img.BBox1DX1, img.BBox1DY1) == (5, 102, 155, 118)

    noreg_vid = by_source(manifest, paths, "b1_vid_noregions").iloc[0]
    assert noreg_vid.RegionSource == "full_frame"
    assert (noreg_vid.BBoxX0, noreg_vid.BBoxY0, noreg_vid.BBoxX1, noreg_vid.BBoxY1) == (0, 0, 160, 120)
    noreg_img = by_source(manifest, paths, "a2_img_noregions").iloc[0]
    assert (noreg_img.BBox2DX1, noreg_img.BBox2DY1) == (160, 120) and pd.isna(noreg_img.BBox1DX0)


def test_region_selection_units():
    big = Region(0, 0, 0, 100, 100, 1, 1)
    inner_color = Region(1, 10, 10, 50, 50, 1, 2)
    other = Region(2, 100, 0, 200, 100, 1, 1)
    dup = Region(3, 100, 0, 200, 100, 1, 1)
    spectral = Region(4, 0, 100, 200, 150, 3, 3)
    assert [r.index for r in select_video_regions([big, inner_color, other, dup, spectral])] == [0, 2]
    two_d, one_d = select_image_regions([inner_color, big, spectral])
    assert two_d.index == 0 and one_d.index == 4
    assert select_image_regions([spectral]) == (None, spectral)


# ---------------------------------------------------------------- naming

def test_output_path_collisions(dicom_tree, make_cfg):
    _, paths = dicom_tree
    cfg = make_cfg()
    run(cfg)
    manifest, _, _ = outputs(cfg)
    name = lambda k: by_source(manifest, paths, k)["RecordID"].iloc[0].rsplit("/", 1)[1]
    assert name("a1_s1_vid") == "2"       # first claimer (series 1 sorts first)
    assert name("a1_s2_vid") == "2_s2"    # same InstanceNumber in series 2
    assert name("a1_s2_noinst") == "na0"  # missing InstanceNumber
    assert manifest["RecordID"].is_unique and manifest["OutputPath"].is_unique
    assert (manifest["OutputPath"] == manifest["RecordID"] + manifest["ItemType"].map({"video": ".avi", "image": ".png"})).all()


def test_naming_is_deterministic(dicom_tree, make_cfg):
    cfg_a, cfg_b, cfg_c = make_cfg("a", chunk_size=4), make_cfg("b", chunk_size=4), make_cfg("c", chunk_size=100)
    for cfg in (cfg_a, cfg_b, cfg_c):
        run(cfg)
    ids = [outputs(c)[0].set_index("SourcePath")["RecordID"].sort_index() for c in (cfg_a, cfg_b, cfg_c)]
    pd.testing.assert_series_equal(ids[0], ids[1])
    # file names inside the study don't depend on the chunk size
    pd.testing.assert_series_equal(ids[0].str.rsplit("/", n=1).str[1], ids[2].str.rsplit("/", n=1).str[1])


def test_collision_across_runs(tmp_path, dicom_tree, make_cfg):
    """A clashing file arriving in a later run gets a suffix; earlier names never change."""
    root, paths = dicom_tree
    cfg = make_cfg()
    run(cfg)
    pairs = lambda m: set(zip(m["SourcePath"], m["RecordID"]))
    before = pairs(outputs(cfg)[0])
    study_a1 = pydicom.dcmread(paths["a1_s1_vid"], stop_before_pixels=True).StudyInstanceUID
    late = synthetic.write(synthetic.make_dataset(patient_id="SYN-A", study_uid=study_a1, series_number=2,
                                                  instance_number=2, n_frames=6), root / "zz_late" / "x.dcm")
    run(cfg)
    after = pairs(outputs(cfg)[0])
    assert before < after
    assert [i for s, i in after - before] == [i for s, i in after if s == str(late.resolve())]
    assert next(iter(after - before))[1].endswith("/2_s2-2")


# ---------------------------------------------------------------- crosswalk

def test_crosswalk_rows_and_patient_reuse(dicom_tree, make_cfg):
    _, paths = dicom_tree
    cfg = make_cfg()
    run(cfg)
    manifest, _, xw = outputs(cfg)
    assert list(xw.columns) == ANON_CROSSWALK_COLUMNS
    assert xw["StudyInstanceUID"].is_unique and len(xw) == 3  # A1, A2, B1
    pat_a = xw[xw["PatientID"] == "SYN-A"]
    assert pat_a["AnonPatientID"].nunique() == 1 and pat_a["DateOffsetDays"].nunique() == 1
    assert sorted(xw["AnonStudyID"]) == ["S000000", "S000001", "S000002"]
    offset = int(pat_a["DateOffsetDays"].iloc[0])
    lo, hi = cfg["index"]["date_offset_days"]
    assert lo <= offset <= hi

    a2 = by_source(manifest, paths, "a2_img_noregions").iloc[0]
    assert a2.PatientAgeCapped == "90+"
    assert a2.StudyDateShifted == shift_date("20210301", offset)
    assert by_source(manifest, paths, "a1_s1_img").iloc[0].PatientAgeCapped == "045Y"


def test_crosswalk_continues_after_restart(tmp_path, dicom_tree, make_cfg):
    root, _ = dicom_tree
    cfg = make_cfg()
    run(cfg)
    xw_path = Path(cfg["paths"]["secure_dir"]) / "anonymization_crosswalk.csv"
    first_bytes = xw_path.read_bytes()
    xw1 = pd.read_csv(xw_path, dtype=str, keep_default_na=False)

    # New run: a new study for an existing patient (in a later chunk) and a brand-new patient.
    synthetic.write(synthetic.make_dataset(patient_id="SYN-A", study_uid="1.2.3.900", n_frames=6), root / "zz" / "a3.dcm")
    synthetic.write(synthetic.make_dataset(patient_id="SYN-D", study_uid="1.2.3.901", n_frames=6), root / "zz" / "d1.dcm")
    run(cfg)
    assert xw_path.read_bytes().startswith(first_bytes)  # append-only, existing rows untouched
    xw2 = pd.read_csv(xw_path, dtype=str, keep_default_na=False)
    new = xw2.iloc[len(xw1):].set_index("StudyInstanceUID")
    old_a = xw1[xw1["PatientID"] == "SYN-A"].iloc[0]
    assert new.loc["1.2.3.900", "AnonPatientID"] == old_a["AnonPatientID"]
    assert new.loc["1.2.3.900", "DateOffsetDays"] == old_a["DateOffsetDays"]
    assert new.loc["1.2.3.901", "AnonPatientID"] == "P000002"
    assert sorted(new["AnonStudyID"]) == ["S000003", "S000004"]

    run(cfg)  # nothing pending -> nothing changes
    assert len(pd.read_csv(xw_path, dtype=str)) == len(xw2)


def test_truncated_crosswalk_line_recovered(dicom_tree, make_cfg):
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

    synthetic.write(synthetic.make_dataset(patient_id="SYN-Z", study_uid="1.2.3.950", n_frames=6), root / "zz" / "z.dcm")
    run(cfg)
    with open(xw_path, newline="") as f:
        rows = list(csv.reader(f))
    assert all(len(r) == len(ANON_CROSSWALK_COLUMNS) for r in rows)
    assert rows[-1][ANON_CROSSWALK_COLUMNS.index("AnonStudyID")] == "S000003"


# ---------------------------------------------------------------- crash / restart

class Boom(Exception):
    pass


@pytest.mark.parametrize("crash_on", ["manifest", "status"])
def test_restart_after_crash(dicom_tree, make_cfg, monkeypatch, crash_on):
    clean = make_cfg("clean")
    run(clean)
    clean_manifest, clean_status, clean_xw = outputs(clean)

    cfg = make_cfg("crashy")
    real_write = script1.atomic_write_parquet
    calls = {"n": 0}

    def flaky_write(df, path):
        # crash on the 2nd chunk, after its crosswalk append, at the chosen write
        if Path(path).parent.name == crash_on and Path(path).name == "chunk-000001.parquet":
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
    pd.testing.assert_series_equal(manifest.set_index("SourcePath")["RecordID"].sort_index(),
                                   clean_manifest.set_index("SourcePath")["RecordID"].sort_index())
    assert len(list((Path(cfg["paths"]["secure_dir"]) / "manifest").glob("*.parquet"))) == \
        len(list((Path(cfg["paths"]["secure_dir"]) / "status").glob("*.parquet")))


def test_multiprocessing_matches_serial(dicom_tree, make_cfg):
    serial, parallel = make_cfg("serial"), make_cfg("parallel", num_workers=2)
    run(serial)
    run(parallel)
    a, b = outputs(serial)[0], outputs(parallel)[0]
    pd.testing.assert_series_equal(a.set_index("SourcePath")["RecordID"].sort_index(),
                                   b.set_index("SourcePath")["RecordID"].sort_index())


# ---------------------------------------------------------------- PHI

def test_no_phi_or_real_keys_in_manifest(dicom_tree, make_cfg):
    cfg = make_cfg()
    run(cfg)
    manifest, status, _ = outputs(cfg)
    forbidden_cols = {"PatientID", "PatientName", "PatientBirthDate", "AccessionNumber", "StudyInstanceUID",
                      "StudyID", "StudyDate", "PatientAge", "InstitutionName", "StudyDescription"}
    assert not forbidden_cols & set(manifest.columns)
    text = manifest.drop(columns=["SourcePath"]).astype(str).to_string()
    for phi in synthetic.FAKE_PHI + ["SYN-A", "SYN-B", "ACCSYN"]:
        assert phi not in text
    assert set(status.columns) == {"SourcePath", "Status", "Detail", "NumRecords"}


def test_age_and_date_helpers():
    assert cap_age("089Y", 90) == "089Y" and cap_age("090Y", 90) == "90+" and cap_age("006M", 90) == "006M"
    assert cap_age("garbage", 90) is None and cap_age("", 90) is None
    assert shift_date("20200101", 1) == "20200102"
    assert shift_date("20200101120000.5", 31) == "20200201120000.5"
    assert shift_date("bad", 5) is None
