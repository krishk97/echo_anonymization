import json
import multiprocessing as mp
import re
import time
from argparse import ArgumentParser
from collections import Counter
from functools import partial
from pathlib import Path

import pandas as pd
from loguru import logger

from src.anon_crosswalk import AnonCrosswalk
from src.io_utils import atomic_write_parquet, load_config, read_parquet_dir, setup_logging
from src.dicom_utils import (cap_age, extract_header, full_frame_region, quiet_pydicom, select_image_regions,
                             select_video_regions, shift_date)


INT_COLUMNS = ["NumFrames", "RegionIndex", "SpatialFormat", "DataType", "DecodedBytes"]
INT_COLUMNS += [f"{p}{c}" for p in ("BBox", "BBox2D", "BBox1D") for c in ("X0", "Y0", "X1", "Y1")]


def find_files_by_extension(folder, extension):
    """Recursively find all files with a given extension.

    Args:
        folder: Path to the directory to search.
        extension: File extension, with or without the leading dot (e.g. "txt" or ".txt").

    Returns:
        A list of Path objects for each matching file.
    """
    extension = extension.lstrip(".").lower()
    return [
        path
        for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() == f".{extension}"
    ]


# ---------------------------------------------------------------------------------------------------
# Output naming
#
# Deterministic output naming: `<AnonPatientID>/<AnonStudyID>/<name>` (RecordID; the extension is added separately).
#
# Rules (first claimer wins; an assigned name never changes):
#   - base name is the InstanceNumber, e.g. `12`
#   - every record of a multi-region video gets `_r<region_index>`: `12_r0`, `12_r1`
#   - a later file whose base is already taken in the study gets `_s<SeriesNumber>` (`_sna` if
#     missing): `12_s3`, then `-<k>` (k = 2, 3, ...) if that is taken too: `12_s3-2`
#   - missing InstanceNumber -> `na<k>`, k = next free number in the study: `na0`, `na1`
#
# The registry holds "stems" (`<AnonPatientID>/<AnonStudyID>/<base>`, i.e. RecordID without `_r<i>`),
# so a file is compared to other files, not to its own regions. It must contain every stem already
# assigned in earlier chunks/runs; with files processed in sorted-path order, names are reproducible.
# ---------------------------------------------------------------------------------------------------
_REGION_SUFFIX = re.compile(r"_r\d+$")


def stem_of(record_id: str) -> str:
    return _REGION_SUFFIX.sub("", record_id)


def assign_stem(anon_patient: str, anon_study: str, instance_number, series_number, registry: set[str]) -> str:
    """Claim and return a free stem for one source file, adding it to `registry`."""
    prefix = f"{anon_patient}/{anon_study}/"
    if instance_number is None:
        k = 0
        while f"{prefix}na{k}" in registry:
            k += 1
        stem = f"{prefix}na{k}"
    else:
        stem = f"{prefix}{instance_number}"
        if stem in registry:
            stem = f"{prefix}{instance_number}_s{'na' if series_number is None else series_number}"
            if stem in registry:
                k = 2
                while f"{stem}-{k}" in registry:
                    k += 1
                stem = f"{stem}-{k}"
    registry.add(stem)
    return stem


def record_id_for(stem: str, region_index: int | None, multi_region: bool) -> str:
    return f"{stem}_r{region_index}" if multi_region else stem


def _bbox_columns(prefix: str, region) -> dict:
    return {f"{prefix}{c.upper()}": (getattr(region, c) if region is not None else None) for c in ("x0", "y0", "x1", "y1")}


def build_records(header: dict, crosswalk: AnonCrosswalk, registry: set[str], cfg: dict) -> list[dict]:
    """Manifest records for one included file (empty if a video has no usable region)."""
    anon_patient, anon_study, offset = crosswalk.lookup(header["crosswalk"]["StudyInstanceUID"])
    regions = header["regions"]
    rows, cols = header["rows"], header["cols"]
    is_video = header["num_frames"] > 1
    region_source = "header" if regions else "full_frame"

    if is_video:
        targets = select_video_regions(regions) if regions else [full_frame_region(rows, cols)]
        if not targets:
            return []
    else:
        two_d, one_d = select_image_regions(regions) if regions else (full_frame_region(rows, cols), None)
        targets = [None]

    stem = assign_stem(anon_patient, anon_study, header["instance_number"], header["series_number"], registry)
    common = {
        "SourcePath": header["source_path"],
        "AnonPatientID": anon_patient,
        "AnonStudyID": anon_study,
        "ItemType": "video" if is_video else "image",
        "SOPClass": header["detail"],
        "NumFrames": header["num_frames"],
        "FPS": header["fps"],
        "RegionSource": region_source,
        "RegionsJSON": json.dumps([r.to_dict() for r in regions]),
        "DecodedBytes": header["decoded_bytes"],
        "Oversize": header["decoded_bytes"] > cfg["max_decoded_gb"] * 1e9,
        "PatientAgeCapped": cap_age(header["patient_age"], cfg["age_cap_years"]),
        **{f"{k}Shifted": shift_date(v, offset) for k, v in header["dates"].items()},
        **header["metadata"],
    }
    records = []
    for region in targets:
        record_id = record_id_for(stem, region.index if region else None, len(targets) > 1)
        rec = {"RecordID": record_id, "OutputPath": record_id + (".avi" if is_video else ".png"), **common}
        if is_video:
            rec.update({"RegionIndex": region.index, **_bbox_columns("BBox", region),
                        "SpatialFormat": region.spatial_format, "DataType": region.data_type,
                        **_bbox_columns("BBox2D", None), **_bbox_columns("BBox1D", None)})
        else:
            rec.update({"RegionIndex": None, **_bbox_columns("BBox", None),
                        "SpatialFormat": None, "DataType": None,
                        **_bbox_columns("BBox2D", two_d), **_bbox_columns("BBox1D", one_d)})
        records.append(rec)
    return records


def process_chunk(headers: list[dict], crosswalk: AnonCrosswalk, registry: set[str], cfg: dict):
    """Single-writer step for one chunk: extend the crosswalk, then build manifest and status rows."""
    included = [h for h in headers if h["status"] == "ok"]
    crosswalk.assign([h["crosswalk"] for h in included])

    manifest_rows, status_rows = [], []
    for h in headers:
        status, n = h["status"], 0
        if status == "ok":
            records = build_records(h, crosswalk, registry, cfg)
            manifest_rows.extend(records)
            n = len(records)
            status = "included" if n else "excluded_no_video_region"
        status_rows.append({"SourcePath": h["source_path"], "Status": status, "Detail": h["detail"], "NumRecords": n})

    manifest = pd.DataFrame(manifest_rows)
    for c in INT_COLUMNS:
        if c in manifest:
            manifest[c] = manifest[c].astype("Int64")
    return manifest, pd.DataFrame(status_rows)


def load_state(manifest_dir: Path, status_dir: Path) -> tuple[set[str], set[str]]:
    """Clean up after a crash and return (done source paths, registry of assigned stems)."""
    for d in (manifest_dir, status_dir):
        d.mkdir(parents=True, exist_ok=True)
        for tmp in d.glob("*.tmp"):
            tmp.unlink()
    committed = {p.name for p in status_dir.glob("*.parquet")}
    for part in manifest_dir.glob("*.parquet"):
        if part.name not in committed:
            logger.warning("Removing uncommitted manifest part {} (no status part)", part.name)
            part.unlink()
    status = read_parquet_dir(status_dir)
    manifest = read_parquet_dir(manifest_dir)
    done = set(status["SourcePath"]) if len(status) else set()
    registry = {stem_of(i) for i in manifest["RecordID"]} if len(manifest) else set()
    return done, registry


def main(cfg: dict) -> Counter:
    paths, icfg = cfg["paths"], cfg["index"]
    quiet_pydicom()
    secure = Path(paths["secure_dir"])
    manifest_dir, status_dir = secure / "manifest", secure / "status"

    crosswalk = AnonCrosswalk(secure / "anonymization_crosswalk.csv", tuple(icfg["date_offset_days"]))
    done, registry = load_state(manifest_dir, status_dir)

    logger.info("Discovering files...")
    files = sorted(str(p) for p in find_files_by_extension(Path(paths["dicom_root"]).resolve(), paths["dicom_extension"]))
    pending = [f for f in files if f not in done]
    chunk_size = icfg["chunk_size"]
    chunks = [pending[i:i + chunk_size] for i in range(0, len(pending), chunk_size)]
    logger.info("{} files found, {} already done, {} pending in {} chunks", len(files), len(files) - len(pending),
                len(pending), len(chunks))

    # Part files are numbered by a global sequence that continues across runs.
    first_seq = 1 + max((int(p.stem.split("-")[1]) for p in status_dir.glob("chunk-*.parquet")), default=-1)
    status_counts, excluded_sop = Counter(), Counter()
    n_records = 0
    worker = partial(extract_header, cfg=icfg)
    pool = mp.Pool(icfg["num_workers"]) if icfg["num_workers"] > 1 else None
    try:
        for k, chunk in enumerate(chunks):
            t0 = time.time()
            headers = pool.map(worker, chunk, chunksize=max(1, len(chunk) // (4 * icfg["num_workers"]))) \
                if pool else [worker(f) for f in chunk]
            manifest, status = process_chunk(headers, crosswalk, registry, icfg)
            name = f"chunk-{first_seq + k:06d}.parquet"
            atomic_write_parquet(manifest, manifest_dir / name)
            atomic_write_parquet(status, status_dir / name)  # commit marker, written last

            status_counts.update(status["Status"])
            excluded_sop.update(status.loc[status["Status"] == "excluded_sop_class", "Detail"])
            n_records += len(manifest)
            logger.info("Chunk {}/{}: {} files -> {} records in {:.1f}s | {}", k + 1, len(chunks), len(chunk),
                        len(manifest), time.time() - t0, dict(Counter(status["Status"])))
    finally:
        if pool:
            pool.close()
            pool.join()

    logger.info("Run summary: {} records written; files by status: {}", n_records, dict(status_counts))
    if excluded_sop:
        logger.info("Excluded by SOP class: {}", dict(excluded_sop))
    logger.info("Crosswalk: {} patients, {} studies; studies seen under a different PatientID: {}",
                len(crosswalk.patients), len(crosswalk.studies), crosswalk.key_conflicts)
    return status_counts


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(Path(config["paths"]["secure_dir"]) / "logs", "index", args.log_level)
    main(config)
