"""DICOM utilities shared by both scripts.

Header side (Script 1, runs in worker processes): header-only reading, SOP class filtering, allowlisted
metadata extraction, date shifting / age capping, and ultrasound region detection and selection.
Workers only read headers and return what they find; they never assign IDs or write files.
Errors are reported as the exception *type* only -- tag values and exception messages can hold PHI.

Pixel side (Script 2): `decode_frames` decodes a file once, frame by frame, into display-ready RGB.

Regions: `find_us_regions` takes a header-only pydicom Dataset and returns a list of `Region`. Everything
downstream (record creation in Script 1, cropping/masking in Script 2) only uses the `Region` fields.
Bounding boxes are half-open pixel ranges for direct numpy slicing: rows [y0, y1), cols [x0, x1).
(DICOM's RegionLocationMaxX1/MaxY1 are inclusive, so `find_us_regions` adds 1.)
"""
import datetime as dt
import json
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pydicom
from pydicom.filereader import read_partial
from pydicom.multival import MultiValue
from pydicom.pixels import iter_pixels
from pydicom.pixels.processing import apply_color_lut
from pydicom.uid import UID_dictionary

from src.anon_crosswalk import PATIENT_FIELDS, STUDY_FIELDS

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


# ---------------------------------------------------------------------------------------------------
# Ultrasound regions
# ---------------------------------------------------------------------------------------------------
# RegionSpatialFormat (0018,6012)
SPATIAL_2D = 1
SPATIAL_MMODE = 2
SPATIAL_SPECTRAL = 3
SPATIAL_FORMAT_NAMES = {0: "None", 1: "2D", 2: "M-Mode", 3: "Spectral", 4: "Wave form", 5: "Graphics"}

# RegionDataType (0018,6014) values kept for video records: unknown/0, Tissue, Color Flow
VIDEO_DATA_TYPES = {None, 0, 1, 2}
ONE_D_SPATIAL_FORMATS = {SPATIAL_MMODE, SPATIAL_SPECTRAL}

# Non-identifying per-region attributes copied into `Region.details`
REGION_DETAIL_TAGS = ["PhysicalUnitsXDirection", "PhysicalUnitsYDirection", "PhysicalDeltaX", "PhysicalDeltaY",
                      "ReferencePixelX0", "ReferencePixelY0", "TransducerFrequency", "PulseRepetitionFrequency"]


@dataclass
class Region:
    index: int  # position in SequenceOfUltrasoundRegions (or 0 for a synthetic full-frame region)
    x0: int
    y0: int
    x1: int
    y1: int
    spatial_format: int | None = None
    data_type: int | None = None
    details: dict = field(default_factory=dict)

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    def contains(self, other: "Region") -> bool:
        return self.x0 <= other.x0 and self.y0 <= other.y0 and self.x1 >= other.x1 and self.y1 >= other.y1

    def to_dict(self) -> dict:
        return {"index": self.index, "x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1,
                "spatial_format": self.spatial_format, "data_type": self.data_type, **self.details}


def _int_or_none(value):
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _primitive(value):
    try:
        return float(value) if isinstance(value, float) else int(value)
    except (TypeError, ValueError):
        return None


def find_us_regions(ds) -> list[Region]:
    """Return the ultrasound regions of a header-only dataset.

    One Region per item of SequenceOfUltrasoundRegions (0018,6011), using
    RegionLocationMinX0/MinY0/MaxX1/MaxY1, RegionSpatialFormat and RegionDataType. Boxes are clipped
    to the image size; items with a missing or empty box are skipped. Returns [] when the sequence is
    absent (callers then fall back to the full frame).
    """
    rows, cols = _int_or_none(ds.get("Rows")), _int_or_none(ds.get("Columns"))
    regions = []
    for i, item in enumerate(ds.get("SequenceOfUltrasoundRegions") or []):
        box = [_int_or_none(item.get(k)) for k in
               ("RegionLocationMinX0", "RegionLocationMinY0", "RegionLocationMaxX1", "RegionLocationMaxY1")]
        if None in box:
            continue
        x0, y0, x1, y1 = box[0], box[1], box[2] + 1, box[3] + 1
        if cols:
            x0, x1 = max(0, min(x0, cols)), max(0, min(x1, cols))
        if rows:
            y0, y1 = max(0, min(y0, rows)), max(0, min(y1, rows))
        if x1 <= x0 or y1 <= y0:
            continue
        details = {k: _primitive(item.get(k)) for k in REGION_DETAIL_TAGS if item.get(k) is not None}
        regions.append(Region(i, x0, y0, x1, y1,
                              _int_or_none(item.get("RegionSpatialFormat")),
                              _int_or_none(item.get("RegionDataType")), details))
    return regions


def full_frame_region(rows: int, cols: int) -> Region:
    return Region(0, 0, 0, cols, rows)


def select_video_regions(regions: list[Region]) -> list[Region]:
    """Regions that become one video record each: 2D spatial format with Tissue/Color Flow/unknown
    data type, minus any region whose box lies fully inside another kept region (e.g. a color box
    inside the 2D sector). For identical boxes the lower index is kept."""
    kept = [r for r in regions if r.spatial_format == SPATIAL_2D and r.data_type in VIDEO_DATA_TYPES]
    return [r for r in kept
            if not any(o is not r and o.contains(r) and (not r.contains(o) or o.index < r.index) for o in kept)]


def select_image_regions(regions: list[Region]) -> tuple[Region | None, Region | None]:
    """(2D region, 1D region) for a single-frame image: the largest 2D region and the largest
    M-mode/Spectral region. Either may be None."""
    two_d = [r for r in regions if r.spatial_format == SPATIAL_2D]
    one_d = [r for r in regions if r.spatial_format in ONE_D_SPATIAL_FORMATS]
    largest = lambda rs: max(rs, key=lambda r: (r.area, -r.index)) if rs else None
    return largest(two_d), largest(one_d)


# ---------------------------------------------------------------------------------------------------
# Header extraction (Script 1)
# ---------------------------------------------------------------------------------------------------
PIXEL_DATA_TAGS = {0x7FE00008, 0x7FE00009, 0x7FE00010}  # Float/DoubleFloat/PixelData, as stop_before_pixels


def quiet_pydicom() -> None:
    """Silence pydicom's log messages and warnings: they can contain file paths and tag values (PHI).
    Call in every process that reads DICOMs (forked workers inherit it)."""
    logging.getLogger("pydicom").disabled = True
    warnings.filterwarnings("ignore")


def read_header(path) -> tuple[pydicom.Dataset, bool]:
    """Read a DICOM header without pixel data. Same as dcmread(stop_before_pixels=True), but also
    reports whether a pixel data element exists (without reading it)."""
    seen = {"pixels": False}

    def stop_at_pixels(tag, vr, length) -> bool:
        if tag in PIXEL_DATA_TAGS:
            seen["pixels"] = True
            return True
        return False

    with open(path, "rb") as fp:
        ds = read_partial(fp, stop_when=stop_at_pixels)
    return ds, seen["pixels"]


def sop_class_name(ds) -> str:
    uid = ds.get("SOPClassUID")
    if not uid:
        return "Missing SOPClassUID"
    return UID_dictionary.get(uid, ("Unknown SOP Class",))[0]


def to_primitive(value):
    """Convert a pydicom element value to a parquet-friendly scalar. Multi-values become a JSON list.
    Anything unconvertible (bytes, sequences, person names, invalid numbers) becomes None."""
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
        if isinstance(value, str) and not isinstance(value, pydicom.valuerep.PersonName):
            return value.strip()
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def compute_fps(ds, default_fps: float) -> float:
    """CineRate -> 1000/FrameTime -> 1000/mean(FrameTimeVector) -> default (as in the old pipeline)."""
    try:
        if ds.get("CineRate") and float(ds.CineRate) > 0:
            return float(ds.CineRate)
        if ds.get("FrameTime") and float(ds.FrameTime) > 0:
            return 1000.0 / float(ds.FrameTime)
        ftv = [float(v) for v in (ds.get("FrameTimeVector") or [])[1:]]  # first entry is always 0
        if ftv and sum(ftv) > 0:
            return 1000.0 / (sum(ftv) / len(ftv))
    except (TypeError, ValueError):
        pass
    return float(default_fps)


def shift_date(value, offset_days: int) -> str | None:
    """Shift a DA (YYYYMMDD) or the date part of a DT value by `offset_days`. Unparseable -> None."""
    if not value:
        return None
    value = str(value).strip()
    try:
        shifted = dt.datetime.strptime(value[:8], "%Y%m%d").date() + dt.timedelta(days=offset_days)
    except (ValueError, OverflowError):
        return None
    return shifted.strftime("%Y%m%d") + value[8:]


def cap_age(value, cap: int) -> str | None:
    """DICOM AS age (e.g. '045Y', '006M') -> same string, or f'{cap}+' when >= cap years. Unparseable -> None."""
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


def _text(ds, keyword) -> str:
    value = ds.get(keyword)
    return "" if value is None else str(value).strip()


def extract_header(path, cfg: dict) -> dict:
    """Worker entry point: everything Script 1 needs from one file's header.

    `cfg` is the `index` section of the config. Returns a dict with `source_path`, `status` and
    `detail`, plus (for status "ok") crosswalk fields, allowlisted metadata, raw dates, regions,
    and geometry. Never raises.
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


# ---------------------------------------------------------------------------------------------------
# Pixel decoding (Script 2)
# ---------------------------------------------------------------------------------------------------
SUPPORTED_PHOTOMETRIC = ["RGB", "YBR_FULL", "YBR_FULL_422", "YBR_RCT", "YBR_ICT", "YBR_PARTIAL_420",
                         "PALETTE COLOR", "MONOCHROME1", "MONOCHROME2"]


def _frame_to_rgb(frame: np.ndarray, photometric: str, ds) -> np.ndarray:
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


def decode_frames(path) -> np.ndarray:
    """Decode all frames of a DICOM file into a (N, H, W, 3) uint8 RGB array.

    Frames are decoded one at a time (pydicom frame-level decoding) into a preallocated array, so a
    compressed clip is never held twice in memory. Raises on unsupported or undecodable pixel data.
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
