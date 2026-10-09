"""DICOM utilities shared by both scripts.

Header side (Script 1, runs in worker processes): header-only reading, SOP class
filtering, allowlisted metadata extraction, date shifting / age capping, and
ultrasound region detection and selection. Workers only read headers and return
what they find; they never assign IDs or write files. Errors are reported as the
exception *type* only -- tag values and exception messages can hold PHI.

Pixel side (Script 2): `decode_frames` decodes a file once, frame by frame, into
display-ready RGB.

Regions: `find_us_regions` takes a header-only pydicom Dataset and returns a list of
`Region`. Everything downstream (record creation in Script 1, cropping/masking in
Script 2) only uses the `Region` fields. Bounding boxes are half-open pixel ranges
for direct numpy slicing: rows [y0, y1), cols [x0, x1). (DICOM's
RegionLocationMaxX1/MaxY1 are inclusive, so `find_us_regions` adds 1.)
"""

import datetime as dt
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.filereader import read_partial
from pydicom.multival import MultiValue
from pydicom.pixels import iter_pixels
from pydicom.pixels.processing import apply_color_lut
from pydicom.uid import UID_dictionary

from src.anon_crosswalk import PATIENT_FIELDS, STUDY_FIELDS

# --------------------------------------------------------------------------------------
# Ultrasound regions
# --------------------------------------------------------------------------------------
# RegionSpatialFormat (0018,6012)
SPATIAL_2D = 1
SPATIAL_MMODE = 2
SPATIAL_SPECTRAL = 3
SPATIAL_FORMAT_NAMES = {
    0: "None",
    1: "2D",
    2: "M-Mode",
    3: "Spectral",
    4: "Wave form",
    5: "Graphics",
}

# RegionDataType (0018,6014)
REGION_DATA_TYPE_DICT = {
    0: None,
    1: "Tissue",
    2: "Color Flow",
    3: "PW Spectral Doppler",
    4: "CW Spectral Doppler",
    5: "Doppler Mean Trace",
    6: "Doppler Mode Trace",
    7: "Doppler Max Trace",
    8: "Volume Trace",
    10: "ECG Trace",
    11: "Pulse Trace",
    12: "Phonocardiogram Trace",
    13: "Gray bar",
    14: "Color bar",
    15: "Integrated Backscatter",
    16: "Area Trace",
    17: "d(area)/dt",
    18: "Other Physiological (Amplitude vs. Time) input",
}

# Data types of a usable 2D region (video records and the 2D box of images):
# unknown/0, Tissue, Color Flow
TWO_D_DATA_TYPES = {None, 0, 1, 2}
ONE_D_SPATIAL_FORMATS = {SPATIAL_MMODE, SPATIAL_SPECTRAL}

# Non-identifying per-region attributes copied into `Region.details`
REGION_DETAIL_TAGS = [
    "PhysicalUnitsXDirection",
    "PhysicalUnitsYDirection",
    "PhysicalDeltaX",
    "PhysicalDeltaY",
    "ReferencePixelX0",
    "ReferencePixelY0",
    "TransducerFrequency",
    "PulseRepetitionFrequency",
]


@dataclass
class Region:
    """One ultrasound region, with a half-open pixel box.

    Attributes:
        index: Position in SequenceOfUltrasoundRegions (or 0 for a synthetic
            full-frame region).
        x0: First column of the box.
        y0: First row of the box.
        x1: One past the last column of the box.
        y1: One past the last row of the box.
        spatial_format: RegionSpatialFormat, or None if missing.
        data_type: RegionDataType, or None if missing.
        details: Non-identifying attributes from `REGION_DETAIL_TAGS`.
    """

    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    spatial_format: int | None = None
    data_type: int | None = None
    details: dict[str, int | float | None] = field(default_factory=dict)

    @property
    def area(self) -> int:
        """Box area in pixels (0 for an empty box)."""
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    def contains(self, other: "Region") -> bool:
        """Returns True if `other`'s box lies fully inside this region's box."""
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )


def _int_or_none(value: Any) -> int | None:
    """Converts a value to int, or None if missing or unconvertible."""
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _primitive(value: Any) -> int | float | None:
    """Converts a value to float (if a float) or int, or None if unconvertible."""
    try:
        return float(value) if isinstance(value, float) else int(value)
    except (TypeError, ValueError):
        return None


def find_us_regions(ds: pydicom.Dataset) -> list[Region]:
    """Returns the ultrasound regions of a header-only dataset.

    One Region per item of SequenceOfUltrasoundRegions (0018,6011), using
    RegionLocationMinX0/MinY0/MaxX1/MaxY1, RegionSpatialFormat and RegionDataType.
    Boxes are clipped to the image size; items with a missing or empty box are
    skipped.

    Args:
        ds: A header-only pydicom dataset.

    Returns:
        The regions in sequence order, or [] when the sequence is absent (callers
        then fall back to the full frame).
    """
    rows, cols = _int_or_none(ds.get("Rows")), _int_or_none(ds.get("Columns"))
    regions = []
    for i, item in enumerate(ds.get("SequenceOfUltrasoundRegions") or []):
        box = [
            _int_or_none(item.get(k))
            for k in (
                "RegionLocationMinX0",
                "RegionLocationMinY0",
                "RegionLocationMaxX1",
                "RegionLocationMaxY1",
            )
        ]
        if None in box:
            continue
        x0, y0, x1, y1 = box[0], box[1], box[2] + 1, box[3] + 1
        if cols:
            x0, x1 = max(0, min(x0, cols)), max(0, min(x1, cols))
        if rows:
            y0, y1 = max(0, min(y0, rows)), max(0, min(y1, rows))
        if x1 <= x0 or y1 <= y0:
            continue
        details = {
            k: _primitive(item.get(k))
            for k in REGION_DETAIL_TAGS
            if item.get(k) is not None
        }
        regions.append(
            Region(
                i,
                x0,
                y0,
                x1,
                y1,
                _int_or_none(item.get("RegionSpatialFormat")),
                _int_or_none(item.get("RegionDataType")),
                details,
            )
        )
    return regions


def full_frame_region(rows: int, cols: int) -> Region:
    """Returns a synthetic region (index 0) covering the whole `rows` x `cols` frame."""
    return Region(0, 0, 0, cols, rows)


def select_video_regions(regions: list[Region]) -> list[Region]:
    """Selects the regions that become one video record each.

    Keeps 2D regions with a Tissue/Color Flow/unknown data type, minus any region
    whose box lies fully inside another kept region (e.g. a color box inside the 2D
    sector). For identical boxes the lower index is kept.

    Args:
        regions: All regions of the file, from `find_us_regions`.

    Returns:
        The selected regions, in input order.
    """
    kept = [
        r
        for r in regions
        if r.spatial_format == SPATIAL_2D and r.data_type in TWO_D_DATA_TYPES
    ]
    return [
        r
        for r in kept
        if not any(
            o is not r and o.contains(r) and (not r.contains(o) or o.index < r.index)
            for o in kept
        )
    ]


def select_image_regions(regions: list[Region]) -> tuple[Region | None, Region | None]:
    """Selects the 2D and 1D regions of a single-frame image.

    Args:
        regions: All regions of the file, from `find_us_regions`.

    Returns:
        (largest 2D Tissue/Color Flow/unknown region, largest M-mode/Spectral
        region). Ties go to the lower index. Either may be None.
    """

    def largest(rs: list[Region]) -> Region | None:
        return max(rs, key=lambda r: (r.area, -r.index)) if rs else None

    two_d = [
        r
        for r in regions
        if r.spatial_format == SPATIAL_2D and r.data_type in TWO_D_DATA_TYPES
    ]
    one_d = [r for r in regions if r.spatial_format in ONE_D_SPATIAL_FORMATS]
    return largest(two_d), largest(one_d)


# --------------------------------------------------------------------------------------
# Header extraction (Script 1)
# --------------------------------------------------------------------------------------
# Float/DoubleFloat/PixelData, as stop_before_pixels
PIXEL_DATA_TAGS = {
    0x7FE00008,
    0x7FE00009,
    0x7FE00010,
}


def quiet_pydicom() -> None:
    """Silences pydicom's log messages and all warnings.

    They can contain file paths and tag values (PHI). Call in every process that
    reads DICOMs (forked workers inherit it).
    """
    logging.getLogger("pydicom").disabled = True
    warnings.filterwarnings("ignore")


def read_header(path: str | Path) -> tuple[pydicom.Dataset, bool]:
    """Reads a DICOM header without pixel data.

    Same as `dcmread(stop_before_pixels=True)`, but also reports whether a pixel
    data element exists (without reading it).

    Args:
        path: DICOM file path.

    Returns:
        (header dataset, whether the file has a pixel data element).
    """
    seen = {"pixels": False}

    def stop_at_pixels(tag: int, vr: str | None, length: int) -> bool:
        if tag in PIXEL_DATA_TAGS:
            seen["pixels"] = True
            return True
        return False

    with open(path, "rb") as fp:
        ds = read_partial(fp, stop_when=stop_at_pixels)
    return ds, seen["pixels"]


def sop_class_name(ds: pydicom.Dataset) -> str:
    """Returns the SOP class name, or "Missing SOPClassUID" / "Unknown SOP Class"."""
    uid = ds.get("SOPClassUID")
    if not uid:
        return "Missing SOPClassUID"
    return UID_dictionary.get(uid, ("Unknown SOP Class",))[0]


def to_primitive(value: Any) -> bool | int | float | str | None:
    """Converts a pydicom element value to a parquet-friendly scalar.

    Args:
        value: An element value, as returned by `Dataset.get`.

    Returns:
        A bool, int, float or stripped str; a JSON list string for a multi-value;
        None for anything unconvertible (bytes, sequences, person names, invalid
        numbers) or empty.
    """
    try:
        if value is None or value == "":
            return None
        if isinstance(value, MultiValue):
            items = [to_primitive(v) for v in value]
            return json.dumps(items)
        if isinstance(value, bool):
            return value
        if isinstance(value, float):  # DSfloat, FD, FL
            return float(value)
        if isinstance(value, int):  # IS, US, UL, SS, SL
            return int(value)
        if isinstance(value, str) and not isinstance(
            value, pydicom.valuerep.PersonName
        ):
            return value.strip()
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def compute_fps(ds: pydicom.Dataset, default_fps: float) -> float:
    """Returns the clip's frame rate.

    Tries CineRate, then 1000/FrameTime, then 1000/mean(FrameTimeVector), as in the
    old pipeline.

    Args:
        ds: Header dataset.
        default_fps: Fallback when none of the tags gives a positive rate.

    Returns:
        Frames per second.
    """
    try:
        if ds.get("CineRate") and float(ds.CineRate) > 0:
            return float(ds.CineRate)
        if ds.get("FrameTime") and float(ds.FrameTime) > 0:
            return 1000.0 / float(ds.FrameTime)
        # The first entry is always 0.
        ftv = [float(v) for v in (ds.get("FrameTimeVector") or [])[1:]]
        if ftv and sum(ftv) > 0:
            return 1000.0 / (sum(ftv) / len(ftv))
    except (TypeError, ValueError):
        pass
    return float(default_fps)


def shift_date(value: Any, offset_days: int) -> str | None:
    """Shifts a DA (YYYYMMDD) or the date part of a DT value.

    Args:
        value: DA or DT value; any time part is kept unchanged.
        offset_days: Days to add.

    Returns:
        The shifted value, or None if `value` is empty or unparseable.
    """
    if not value:
        return None
    value = str(value).strip()
    try:
        shifted = dt.datetime.strptime(value[:8], "%Y%m%d").date() + dt.timedelta(
            days=offset_days
        )
    except (ValueError, OverflowError):
        return None
    return shifted.strftime("%Y%m%d") + value[8:]


def cap_age(value: Any, cap: int) -> str | None:
    """Caps a DICOM AS age (e.g. '045Y', '006M').

    Args:
        value: AS age string.
        cap: Age in years at or above which the age is replaced.

    Returns:
        The stripped age, f"{cap}+" when it is `cap` years or more, or None if
        empty or unparseable.
    """
    if not value:
        return None
    value = str(value).strip()
    try:
        number, unit = int(value[:-1]), value[-1].upper()
    except (ValueError, IndexError):
        return None
    if unit not in "DWMY":
        return None
    years = number / {"D": 365.25, "W": 52.18, "M": 12, "Y": 1}[unit]
    return f"{cap}+" if years >= cap else value


def _text(ds: pydicom.Dataset, keyword: str) -> str:
    """Returns the element's value as a stripped string ("" if missing)."""
    value = ds.get(keyword)
    return "" if value is None else str(value).strip()


def extract_header(path: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Reads everything Script 1 needs from one file's header (worker entry point).

    Never raises: any exception becomes status "error" with the exception type
    as detail.

    Args:
        path: DICOM file path.
        cfg: The `index` section of the config.

    Returns:
        A dict with `source_path`, `status` and `detail`, plus (for status "ok")
        crosswalk fields, allowlisted metadata, raw dates, regions, and geometry.
    """
    out = {"source_path": str(path), "status": "error", "detail": None}
    try:
        ds, has_pixels = read_header(path)
        sop_name = sop_class_name(ds)
        if sop_name not in cfg["sop_classes"]:
            return {**out, "status": "excluded_sop_class", "detail": sop_name}
        if not has_pixels:
            return {**out, "status": "excluded_no_pixel_data", "detail": sop_name}
        if not _text(ds, "PatientID"):
            return {**out, "status": "excluded_missing_patient_id", "detail": sop_name}
        if not _text(ds, "StudyInstanceUID"):
            return {**out, "status": "excluded_missing_study_uid", "detail": sop_name}

        rows, cols = int(ds.Rows), int(ds.Columns)
        num_frames = int(ds.get("NumberOfFrames") or 1)
        if cfg["max_num_frames"] is not None and num_frames > cfg["max_num_frames"]:
            return {**out, "status": "excluded_too_many_frames", "detail": sop_name}
        bytes_per_sample = (int(ds.get("BitsAllocated") or 8) + 7) // 8
        return {
            **out,
            "status": "ok",
            "detail": sop_name,
            "crosswalk": {k: _text(ds, k) for k in PATIENT_FIELDS + STUDY_FIELDS},
            "metadata": {k: to_primitive(ds.get(k)) for k in cfg["metadata_allowlist"]},
            "dates": {k: _text(ds, k) for k in cfg["date_fields"]},
            "patient_age": _text(ds, "PatientAge"),
            "instance_number": to_primitive(ds.get("InstanceNumber")),
            "series_number": to_primitive(ds.get("SeriesNumber")),
            "rows": rows,
            "cols": cols,
            "num_frames": num_frames,
            "fps": compute_fps(ds, cfg["default_fps"]),
            # Decoded size after conversion to 3-channel RGB
            "decoded_bytes": num_frames * rows * cols * 3 * bytes_per_sample,
            "regions": find_us_regions(ds),
        }
    except Exception as e:  # never let one bad file crash the run; record the type only
        return {**out, "status": "error", "detail": type(e).__name__}


# --------------------------------------------------------------------------------------
# Pixel decoding (Script 2)
# --------------------------------------------------------------------------------------
SUPPORTED_PHOTOMETRIC = [
    "RGB",
    "YBR_FULL",
    "YBR_FULL_422",
    "YBR_RCT",
    "YBR_ICT",
    "YBR_PARTIAL_420",
    "PALETTE COLOR",
    "MONOCHROME1",
    "MONOCHROME2",
]


def _frame_to_rgb(
    frame: np.ndarray, photometric: str, ds: pydicom.Dataset
) -> np.ndarray:
    """Converts one decoded frame to (H, W, 3) uint8 RGB.

    Raises:
        ValueError: The pixels are not 8-bit after any palette lookup.
    """
    if photometric == "PALETTE COLOR":
        frame = apply_color_lut(frame, ds)
        if frame.dtype == np.uint16:  # LUTs are often 16-bit; scale to 8-bit
            frame = (frame >> 8).astype(np.uint8)
    if frame.dtype != np.uint8:
        raise ValueError("Decoded pixels are not 8-bit")
    if photometric == "MONOCHROME1":
        frame = np.stack([255 - frame] * 3, axis=-1)
    elif photometric == "MONOCHROME2":
        frame = np.stack([frame] * 3, axis=-1)
    # RGB / YBR_*: iter_pixels already returns RGB (as_rgb=True by default)
    return frame


def decode_frames(path: str | Path) -> np.ndarray:
    """Decodes all frames of a DICOM file into a (N, H, W, 3) uint8 RGB array.

    Frames are decoded one at a time (pydicom frame-level decoding) into a
    preallocated array, so a compressed clip is never held twice in memory.

    Args:
        path: DICOM file path.

    Returns:
        The decoded frames; N is at most NumberOfFrames.

    Raises:
        ValueError: Unsupported PhotometricInterpretation, non-8-bit pixels, or no
            frames decoded. pydicom may raise other errors for undecodable data.
    """
    ds, _ = read_header(path)
    photometric = str(ds.get("PhotometricInterpretation", "")).strip()
    if photometric not in SUPPORTED_PHOTOMETRIC:
        raise ValueError("Unsupported PhotometricInterpretation")
    n_frames = int(ds.get("NumberOfFrames") or 1)
    out = np.empty((n_frames, int(ds.Rows), int(ds.Columns), 3), dtype=np.uint8)
    n = 0
    for frame in iter_pixels(path):
        if n >= n_frames:
            break
        out[n] = _frame_to_rgb(frame, photometric, ds)
        n += 1
    if n == 0:
        raise ValueError("No frames decoded")
    return out[:n]
