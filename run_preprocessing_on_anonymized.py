"""Preprocesses an already-anonymized DICOM dataset (e.g. MIMIC-IV-ECHO).

For datasets that were de-identified elsewhere and come with an echo record list
(one row per DICOM, with its path and patient/study IDs) and optionally a study list.
No crosswalk, ID assignment or date shift: the IDs come from the record list and the
header dates are copied unchanged (into the usual `<Date>Shifted` columns).

Steps:
  1. Read every listed DICOM header (as Script 1) and write a work manifest to
     `--work-dir` (it holds source paths, so keep it outside `--output-root`). An
     existing work manifest is reused, so a crashed run resumes without rereading
     headers; delete the work dir to rebuild it.
  2. Run the standard preprocessing (crop, mask, save media) on CPU, or with
     `--classify` also the view classifiers on GPU, then `merge`.
  3. Add `SourceFilePath` (the record list's path value) to the video/image lists and
     write them in the record list's format (CSV or parquet). A given study list is
     filtered to the studies with records and written in its own format.
"""

import multiprocessing as mp
import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

import run_echo_preprocessing_and_anonymization as preprocess_only
import run_echo_preprocessing_and_anonymization_with_classifier as with_classifier
from create_anonymization_plan import process_chunk
from src.dicom_utils import extract_header, quiet_pydicom
from src.io_utils import (
    atomic_write_parquet,
    load_config,
    read_parquet_dir,
    setup_logging,
)

ITEM_TYPES = ("video", "image")
MANIFEST_PART = "chunk-000000.parquet"


class ListIds:
    """Stands in for `AnonCrosswalk` in `process_chunk`: IDs from the record list.

    Each header's StudyInstanceUID is replaced by its source path beforehand, so
    `lookup` returns that file's record list IDs and a date offset of 0.

    Attributes:
        ids: (PatientID, StudyID) per source path.
    """

    def __init__(self, ids: dict[str, tuple[str, str]]) -> None:
        """Initializes the lookup.

        Args:
            ids: (PatientID, StudyID) per source path.
        """
        self.ids = ids

    def assign(self, keys: list[dict[str, str]]) -> None:
        """Does nothing: the IDs already exist.

        Args:
            keys: The included headers' crosswalk fields (unused).
        """

    def lookup(self, source_path: str) -> tuple[str, str, int]:
        """Returns (PatientID, StudyID, date offset 0) of one file.

        Args:
            source_path: The file's source path (in place of its StudyInstanceUID).

        Returns:
            The record list's PatientID and StudyID, and a date offset of 0 days.
        """
        return *self.ids[source_path], 0


def _is_csv(path: Path) -> bool:
    """Returns True for a CSV list (`.csv` or `.csv.gz`), False for parquet."""
    return ".csv" in Path(path).suffixes


def read_list(path: Path) -> pd.DataFrame:
    """Reads a CSV (as text, so values are written back unchanged) or parquet list."""
    if _is_csv(path):
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.read_parquet(path)


def write_list(df: pd.DataFrame, path: Path) -> None:
    """Writes a list atomically, as CSV or parquet depending on the extension."""
    if path.suffix == ".parquet":
        atomic_write_parquet(df, path)
    else:
        tmp = path.with_name(path.name + ".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)


def build_work_dir(
    records: pd.DataFrame, dicom_root: Path, work_dir: Path, cfg: dict[str, Any]
) -> None:
    """Reads the listed headers and writes the work manifests, status and crosswalk.

    Args:
        records: One row per listed file: SourceFilePath, PatientID, StudyID.
        dicom_root: Directory the listed paths are relative to.
        work_dir: Output directory (secure_dir of the derived config).
        cfg: The `index` section of the config.
    """
    records = records.drop_duplicates("SourceFilePath").sort_values("SourceFilePath")
    paths = [str(dicom_root / p.lstrip("/")) for p in records["SourceFilePath"]]
    id_pairs = records[["PatientID", "StudyID"]].itertuples(index=False, name=None)
    ids = dict(zip(paths, id_pairs, strict=True))
    logger.info("Reading {} headers...", len(paths))
    worker = partial(extract_header, cfg=cfg)
    if cfg["num_workers"] > 1:
        with mp.Pool(cfg["num_workers"]) as pool:
            chunksize = max(1, len(paths) // (4 * cfg["num_workers"]))
            headers = pool.map(worker, paths, chunksize=chunksize)
    else:
        headers = [worker(p) for p in paths]

    study_times = {}
    for h in headers:
        if h["status"] == "ok":
            h["crosswalk"]["StudyInstanceUID"] = h["source_path"]
            study_id = ids[h["source_path"]][1]
            study_times.setdefault(study_id, h["crosswalk"]["StudyTime"])
    videos, images, status = process_chunk(headers, ListIds(ids), set(), cfg)

    source_file = dict(zip(paths, records["SourceFilePath"], strict=True))
    for item_type, manifest in zip(ITEM_TYPES, (videos, images), strict=True):
        if len(manifest):
            manifest["SourceFilePath"] = manifest["SourcePath"].map(source_file)
        atomic_write_parquet(
            manifest, work_dir / "manifest" / item_type / MANIFEST_PART
        )
    # Only the two columns `merge` reads to build the study list.
    pd.DataFrame(
        {"AnonStudyID": list(study_times), "StudyTime": list(study_times.values())}
    ).to_csv(work_dir / "anonymization_crosswalk.csv", index=False)
    # Commit marker, written last.
    atomic_write_parquet(status, work_dir / "status" / MANIFEST_PART)
    logger.info(
        "Work manifest: {} video + {} image records | files by status: {}",
        len(videos),
        len(images),
        status["Status"].value_counts().to_dict(),
    )


def finalize_outputs(
    tables: tuple[pd.DataFrame, pd.DataFrame],
    cfg: dict[str, Any],
    ext: str,
    study_list: Path | None,
    study_column: str,
) -> None:
    """Adds SourceFilePath and writes the final lists in the input formats.

    Args:
        tables: (video table, image table) from `merge`.
        cfg: The derived config.
        ext: ".csv" or ".parquet", the record list's format.
        study_list: The input study list, or None to keep merge's study list.
        study_column: The study list's study ID column.
    """
    out_root = Path(cfg["paths"]["output_root"])
    manifests = with_classifier.read_manifests(cfg)
    study_ids = set()
    for item_type, table in zip(ITEM_TYPES, tables, strict=True):
        if len(table):
            source_file = manifests[item_type].set_index("RecordID")["SourceFilePath"]
            table["SourceFilePath"] = table["RecordID"].map(source_file)
            study_ids |= set(table["StudyID"])
        parquet = out_root / f"echo-{item_type}-list.parquet"
        write_list(table, out_root / f"echo-{item_type}-list{ext}")
        if ext != ".parquet":
            parquet.unlink(missing_ok=True)

    if study_list is not None:
        studies = read_list(study_list)
        studies = studies[studies[study_column].astype(str).isin(study_ids)]
        study_ext = ".csv" if _is_csv(study_list) else ".parquet"
        write_list(studies, out_root / f"echo-study-list{study_ext}")
        if study_ext != ".csv":
            (out_root / "echo-study-list.csv").unlink(missing_ok=True)
        logger.info("Study list: {} studies (from {})", len(studies), study_list.name)


def main(
    cfg: dict[str, Any],
    record_list: Path,
    output_root: Path,
    work_dir: Path,
    study_list: Path | None = None,
    path_column: str = "FilePath",
    patient_column: str = "PatientID",
    study_column: str = "StudyID",
    dicom_root: Path | None = None,
    classify: bool = False,
    retry_errors: bool = False,
    log_level: str = "INFO",
) -> bool:
    """Preprocesses the listed DICOMs and writes the final lists.

    Args:
        cfg: The config (`index`, `preprocess`, `models`); any `paths` section is
            replaced by `work_dir`/`output_root`.
        record_list: The record list (CSV or parquet).
        output_root: Output directory (media, results, final lists).
        work_dir: Directory for the work manifest (contains source paths).
        study_list: Optional study list (CSV or parquet) to filter and copy.
        path_column: The record list's DICOM path column.
        patient_column: The record list's patient ID column.
        study_column: The study ID column of the record and study lists.
        dicom_root: Directory the listed paths are relative to (default: the record
            list's directory).
        classify: Also run the view classifiers (GPU).
        retry_errors: Re-queue records whose latest status is `error`.
        log_level: Log level for the spawned processes.

    Returns:
        False if a preprocessing process failed (nothing merged), else True.

    Raises:
        ValueError: A column is missing from the record list.
    """
    cfg["paths"] = {"secure_dir": str(work_dir), "output_root": str(output_root)}
    if read_parquet_dir(work_dir / "status").empty:
        columns = {path_column: "SourceFilePath"}
        columns.update({patient_column: "PatientID", study_column: "StudyID"})
        records = read_list(record_list)
        missing = [c for c in columns if c not in records.columns]
        if missing:
            raise ValueError(f"Record list has no column(s) {missing}")
        records = records[list(columns)].rename(columns=columns).astype(str)
        root = dicom_root if dicom_root is not None else record_list.parent
        build_work_dir(records, Path(root), work_dir, cfg["index"])
    else:
        logger.info("Reusing the work manifest in {}", work_dir)

    run = with_classifier.run if classify else preprocess_only.run
    if not run(cfg, retry_errors, log_level):
        return False
    tables = with_classifier.merge(cfg)
    if any(len(t) for t in tables):
        ext = ".csv" if _is_csv(record_list) else ".parquet"
        finalize_outputs(tables, cfg, ext, study_list, study_column)
    return True


def cli() -> None:
    """Parses the command line and runs `main`."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to the YAML config"
    )
    parser.add_argument("--record-list", type=Path, required=True)
    parser.add_argument("--study-list", type=Path)
    parser.add_argument("--path-column", default="FilePath")
    parser.add_argument("--patient-column", default="PatientID")
    parser.add_argument("--study-column", default="StudyID")
    parser.add_argument(
        "--dicom-root",
        type=Path,
        help="Directory the listed paths are relative to (default: the record "
        "list's directory)",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Work manifest directory (holds source paths; keep it outside "
        "--output-root)",
    )
    parser.add_argument(
        "--classify", action="store_true", help="Also run the view classifiers (GPU)"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reprocess records whose latest status is error",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.output_root / "logs", "preprocess_anonymized", args.log_level)
    quiet_pydicom()
    ok = main(
        load_config(args.config),
        args.record_list,
        args.output_root,
        args.work_dir,
        study_list=args.study_list,
        path_column=args.path_column,
        patient_column=args.patient_column,
        study_column=args.study_column,
        dicom_root=args.dicom_root,
        classify=args.classify,
        retry_errors=args.retry_errors,
        log_level=args.log_level,
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
