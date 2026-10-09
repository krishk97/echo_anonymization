"""Synthetic DICOM fixture generator.

All data is fake; nothing here touches real files. Each file carries some fake
"PHI" (names, institution, descriptions, a private tag) so tests can check that
nothing outside the allowlist leaks into outputs.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

US_IMAGE = "1.2.840.10008.5.1.4.1.1.6.1"
US_MULTIFRAME = "1.2.840.10008.5.1.4.1.1.3.1"
CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2"

FAKE_PHI = [
    "SYNTH^PATIENT",
    "Synthetic General Hospital",
    "SECRET STUDY DESCRIPTION",
    "DR^OPERATOR",
    "PRIVATE-PHI",
]

ROWS, COLS = 120, 160
# (x0, y0, x1_inclusive, y1_inclusive, RegionSpatialFormat, RegionDataType)
RegionSpec = tuple[int, int, int, int, int, int]
SECTOR = (20, 10, 109, 99, 1, 1)  # 2D tissue
DOPPLER_STRIP = (5, 102, 154, 117, 3, 3)  # spectral (PW) strip
LEFT, RIGHT = (5, 10, 74, 99, 1, 1), (85, 10, 154, 99, 1, 1)
COLOR_BOX_IN_LEFT = (20, 30, 60, 80, 1, 2)  # color flow box inside LEFT -> deduplicated
ECG_STRIP = (0, 105, 159, 119, 4, 10)  # waveform -> never a record
GRAY_BAR_2D = (150, 10, 159, 99, 1, 13)  # 2D format but not tissue/color -> unusable


def _regions_seq(regions: Iterable[RegionSpec]) -> Sequence:
    """Builds a SequenceOfUltrasoundRegions from region specs."""
    seq = []
    for x0, y0, x1, y1, sf, dt in regions:
        item = Dataset()
        item.RegionSpatialFormat = sf
        item.RegionDataType = dt
        item.RegionFlags = 0
        item.RegionLocationMinX0, item.RegionLocationMinY0 = x0, y0
        item.RegionLocationMaxX1, item.RegionLocationMaxY1 = x1, y1
        item.PhysicalUnitsXDirection = item.PhysicalUnitsYDirection = 3
        item.PhysicalDeltaX = item.PhysicalDeltaY = 0.05
        seq.append(item)
    return Sequence(seq)


def _pixels(
    n_frames: int, regions: Iterable[RegionSpec] | None, seed: int
) -> np.ndarray:
    """Makes fake RGB frames of shape (n_frames, ROWS, COLS, 3).

    A static 'burned-in text' bar sits outside the regions, and moving noise fills
    a disc in each non-waveform region (or in the whole frame if `regions` is
    None).
    """
    rng = np.random.default_rng(seed)
    px = np.zeros((n_frames, ROWS, COLS, 3), dtype=np.uint8)
    px[:, 0:6, :, :] = 220  # fake header text bar (static)
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    for x0, y0, x1, y1, sf, _ in regions or [(0, 6, COLS - 1, ROWS - 1, 1, 1)]:
        if sf == 4:
            continue
        cy, cx, r = (y0 + y1) / 2, (x0 + x1) / 2, min(x1 - x0, y1 - y0) / 2 - 2
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        noise = rng.integers(40, 255, size=(n_frames, ROWS, COLS, 1), dtype=np.uint8)
        px[:, disc] = noise[:, disc]
    return px


def make_dataset(
    *,
    patient_id: str | None = "SYN0001",
    study_uid: str | None = None,
    series_number: int = 1,
    instance_number: int | None = 1,
    n_frames: int = 1,
    regions: tuple[RegionSpec, ...] = (SECTOR,),
    sop_class: str | None = None,
    with_pixels: bool = True,
    patient_age: str = "045Y",
    study_date: str = "20200115",
    seed: int = 0,
    include_regions: bool = True,
) -> Dataset:
    """Builds one synthetic ultrasound (or CT) dataset with fake PHI.

    Args:
        patient_id: PatientID, or None to leave it out.
        study_uid: StudyInstanceUID, or None to leave it out.
        series_number: SeriesNumber.
        instance_number: InstanceNumber, or None to leave it out.
        n_frames: Number of frames; more than 1 makes a multi-frame clip.
        regions: Region specs for SequenceOfUltrasoundRegions and the pixels.
        sop_class: SOP class UID; defaults to US image or US multi-frame.
        with_pixels: Whether to add PixelData.
        patient_age: PatientAge (AS).
        study_date: StudyDate and the other DA dates (YYYYMMDD).
        seed: Seed for the pixel noise.
        include_regions: Whether to add SequenceOfUltrasoundRegions.

    Returns:
        The dataset, with file meta, ready for `write`.
    """
    sop_class = sop_class or (US_MULTIFRAME if n_frames > 1 else US_IMAGE)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
    ds.Modality = "US" if sop_class != CT_IMAGE else "CT"
    if patient_id is not None:
        ds.PatientID = patient_id
    ds.PatientName = FAKE_PHI[0]
    ds.PatientSex = "F"
    ds.PatientAge = patient_age
    ds.PatientBirthDate = "19700101"
    ds.InstitutionName = FAKE_PHI[1]
    ds.StudyDescription = FAKE_PHI[2]
    ds.OperatorsName = FAKE_PHI[3]
    ds.add_new(0x00091010, "LO", FAKE_PHI[4])  # private tag
    if study_uid is not None:
        ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.AccessionNumber = "ACC" + patient_id if patient_id else "ACC"
    ds.StudyID = "1"
    ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = ds.ContentDate = study_date
    ds.StudyTime = "101500"
    ds.AcquisitionDateTime = study_date + "101730.250000"
    ds.SeriesNumber = series_number
    if instance_number is not None:
        ds.InstanceNumber = instance_number
    ds.Manufacturer, ds.ManufacturerModelName = "SynthCorp", "EchoSim 1"

    ds.Rows, ds.Columns = ROWS, COLS
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    if n_frames > 1:
        ds.NumberOfFrames = n_frames
        ds.FrameTime = 40.0  # 25 fps
    if include_regions and regions:
        ds.SequenceOfUltrasoundRegions = _regions_seq(regions)
    if with_pixels:
        px = _pixels(n_frames, regions if include_regions else None, seed)
        ds.PixelData = (px if n_frames > 1 else px[0]).tobytes()
    return ds


def write(ds: Dataset, path: str | Path) -> Path:
    """Saves `ds` as a DICOM file at `path`, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path


def build_tree(root: str | Path) -> dict[str, str]:
    """Writes the standard synthetic dataset under `root`.

    Files are named so sorted-path order is deterministic (series 1 sorts before
    series 2).

    Args:
        root: Directory to write into.

    Returns:
        {name: resolved path} for assertions.
    """
    root = Path(root)
    study_a1, study_a2, study_b1 = generate_uid(), generate_uid(), generate_uid()
    files = {
        # Patient A, study A1, series 1
        "a1_s1_img": make_dataset(
            patient_id="SYN-A",
            study_uid=study_a1,
            instance_number=1,
            regions=(SECTOR, DOPPLER_STRIP, ECG_STRIP),
        ),
        "a1_s1_vid": make_dataset(
            patient_id="SYN-A",
            study_uid=study_a1,
            instance_number=2,
            n_frames=12,
            seed=1,
        ),
        "a1_s1_multi": make_dataset(
            patient_id="SYN-A",
            study_uid=study_a1,
            instance_number=3,
            n_frames=12,
            seed=2,
            regions=(LEFT, RIGHT, COLOR_BOX_IN_LEFT, ECG_STRIP),
        ),
        # Patient A, study A1, series 2: InstanceNumber 2 clashes with series 1; one
        # missing InstanceNumber
        "a1_s2_vid": make_dataset(
            patient_id="SYN-A",
            study_uid=study_a1,
            series_number=2,
            instance_number=2,
            n_frames=10,
            seed=3,
        ),
        "a1_s2_noinst": make_dataset(
            patient_id="SYN-A",
            study_uid=study_a1,
            series_number=2,
            instance_number=None,
            n_frames=10,
            seed=4,
        ),
        # Patient A, study A2 (later study, same patient): image, age >= 90
        "a2_img": make_dataset(
            patient_id="SYN-A",
            study_uid=study_a2,
            instance_number=1,
            patient_age="093Y",
            study_date="20210301",
        ),
        # Patient B: video without a region sequence; Doppler-only clip (no 2D region)
        "b1_vid_noregions": make_dataset(
            patient_id="SYN-B",
            study_uid=study_b1,
            instance_number=1,
            n_frames=8,
            include_regions=False,
            seed=5,
        ),
        "b1_doppler_only": make_dataset(
            patient_id="SYN-B",
            study_uid=study_b1,
            instance_number=2,
            n_frames=8,
            regions=(DOPPLER_STRIP,),
            seed=6,
        ),
        # Excluded images in study B1: no region sequence; only unusable regions
        "x_img_noregions": make_dataset(
            patient_id="SYN-B",
            study_uid=study_b1,
            instance_number=3,
            include_regions=False,
        ),
        "x_img_unusable_regions": make_dataset(
            patient_id="SYN-B",
            study_uid=study_b1,
            instance_number=4,
            regions=(ECG_STRIP, GRAY_BAR_2D),
        ),
        # Excluded files
        "x_ct": make_dataset(
            patient_id="SYN-C", study_uid=generate_uid(), sop_class=CT_IMAGE
        ),
        "x_nopixels": make_dataset(
            patient_id="SYN-C", study_uid=generate_uid(), with_pixels=False
        ),
        "x_no_patient_id": make_dataset(patient_id=None, study_uid=generate_uid()),
        "x_no_study_uid": make_dataset(patient_id="SYN-C", study_uid=None),
    }
    layout = {
        "a1_s1_img": "patA/studyA1/s1_i1.dcm",
        "a1_s1_vid": "patA/studyA1/s1_i2.dcm",
        "a1_s1_multi": "patA/studyA1/s1_i3.dcm",
        "a1_s2_vid": "patA/studyA1/s2_i2.dcm",
        "a1_s2_noinst": "patA/studyA1/s2_noinst.dcm",
        "a2_img": "patA/studyA2/i1.dcm",
        "b1_vid_noregions": "patB/studyB1/i1.dcm",
        "b1_doppler_only": "patB/studyB1/i2.dcm",
        "x_img_noregions": "patB/studyB1/i3.dcm",
        "x_img_unusable_regions": "patB/studyB1/i4.dcm",
        "x_ct": "other/ct.dcm",
        "x_nopixels": "other/nopixels.dcm",
        "x_no_patient_id": "other/no_pid.dcm",
        "x_no_study_uid": "other/no_suid.dcm",
    }
    paths = {name: write(ds, root / layout[name]) for name, ds in files.items()}
    corrupt = root / "other" / "corrupt.dcm"
    corrupt.write_bytes(b"\x00" * 128 + b"DICM" + b"\xff" * 40)
    paths["x_corrupt"] = corrupt
    # Wrong extension, never discovered.
    (root / "other" / "notes.txt").write_text("not a dicom")
    return {k: str(Path(v).resolve()) for k, v in paths.items()}
