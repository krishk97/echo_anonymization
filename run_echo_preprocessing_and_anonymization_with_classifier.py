"""Script 2 (GPU): decode, crop/mask, classify (EchoFine + EchoPrime) and save every pending manifest record.

Subcommands:
  run    process pending records with one spawned process per GPU
  merge  combine result parts into the final anonymized results table

Restartable: the done set is every RecordID already in a result part (any status); `--retry-errors`
re-queues records whose latest status is `error`. A result row is written only after both models
have run and the media file has been saved atomically. Records saved by `run_echo_preprocessing_and_anonymization.py` (status
`preprocessed`, no view columns) are not done here: `run` redoes them with classification.

Model inputs (per source file / record):
  - EchoFine:  the full, unmasked frames (as in the old pipeline), run once per source file and
               shared by all of that file's region records.
  - EchoPrime: the record's preprocessed (cropped + PHI-masked) media.
Only the preprocessed media is written to disk.
"""
import multiprocessing as mp
import os
import shutil
import time
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.dicom_utils import decode_frames, quiet_pydicom
from src.image_utils import crop_and_scale, mask_outside_ultrasound_video, save_video_cv2
from src.io_utils import atomic_write_parquet, load_config, read_parquet_dir, setup_logging
from src.view_classifier.echoprime import ECHOPRIME_VIEWS, classify_view_torch, setup_echoprime_view_classifier


MODEL_RESOLUTION = (224, 224)
# Manifest columns passed to the GPU processes (everything else stays in the manifest)
BBOX_COLUMNS = [f"{p}{c}" for p in ("BBox", "BBox2D", "BBox1D") for c in ("X0", "Y0", "X1", "Y1")]
RECORD_FIELDS = ["RecordID", "OutputPath", "ItemType", "FPS", "Oversize"] + BBOX_COLUMNS
# Manifest columns copied into the final table (besides the config allowlist and shifted dates).
# This is a keep-list: SourcePath and anything else not listed here never reach the final table.
MANIFEST_KEEP = ["RecordID", "AnonPatientID", "AnonStudyID", "OutputPath", "ItemType", "SOPClass", "NumFrames", "FPS",
                 "RegionSource", "RegionIndex", "SpatialFormat", "DataType", "RegionsJSON", "DecodedBytes",
                 "Oversize", "PatientAgeCapped"] + BBOX_COLUMNS
RESULT_KEEP = ["Status", "ErrorType", "PipelineVersion", "Timestamp"]
# Prefixes of the view-classification columns written by summarize()
MODEL_COL_PREFIXES = ("EchoFine_", "EchoPrime_", "View_", "ViewScore_")


# ---------------------------------------------------------------------------------------------------
# Data loading (runs in DataLoader worker processes)
# ---------------------------------------------------------------------------------------------------
def _bbox(rec: dict, prefix: str):
    """(x0, y0, x1, y1) from a manifest record, or None if absent."""
    box = tuple(rec[f"{prefix}{c}"] for c in ("X0", "Y0", "X1", "Y1"))
    return None if box[0] is None else tuple(int(v) for v in box)


def resize_frames(frames: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) -> (N, 224, 224, 3), as in the old pipeline (crop to aspect ratio, no zoom)."""
    return np.stack([crop_and_scale(f, res=MODEL_RESOLUTION, zoom=0.0) for f in frames], axis=0)


class AnonymizedEchoDataset(Dataset):
    """One item = one source file with all of its pending records, so a file is decoded only once.

    Returns {"echofine_in": (N,224,224,3) or None, "records": [...]}, where each record dict has
    RecordID/OutputPath/ItemType/FPS/Status/ErrorType and, when Status is "ok", the preprocessed
    `media` to save and the `echoprime_in` model input. Never raises: failures become skip/error records.
    """

    def __init__(self, files: list[tuple[str, list[dict]]]):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        source_path, records = self.files[index]
        base = [{k: r[k] for k in ("RecordID", "OutputPath", "ItemType", "FPS")} for r in records]
        mark = lambda status, err=None: [{**b, "Status": status, "ErrorType": err} for b in base]

        if records[0]["Oversize"]:
            return {"echofine_in": None, "records": mark("skipped_oversize")}
        try:
            frames = decode_frames(source_path)
        except Exception as e:
            return {"echofine_in": None, "records": mark("skipped_undecodable", type(e).__name__)}
        try:
            echofine_in = resize_frames(frames)
        except Exception as e:
            return {"echofine_in": None, "records": mark("error", type(e).__name__)}

        out = []
        for rec, b in zip(records, base):
            try:
                if rec["ItemType"] == "video":
                    media = self.preprocess_echo_video(frames, _bbox(rec, "BBox"))
                    echoprime_in = resize_frames(media)
                else:
                    media = self.preprocess_echo_image(frames[0], _bbox(rec, "BBox2D"), _bbox(rec, "BBox1D"))
                    echoprime_in = resize_frames(media[np.newaxis])
                out.append({**b, "Status": "ok", "ErrorType": None, "media": media, "echoprime_in": echoprime_in})
            except Exception as e:
                out.append({**b, "Status": "error", "ErrorType": type(e).__name__})
        return {"echofine_in": echofine_in, "records": out}

    @staticmethod
    def preprocess_echo_video(video: np.ndarray, bbox: tuple) -> np.ndarray:
        """Burned-in PHI removal for one video region.

        Crops to the region's bounding box, then zeroes stationary pixels (text, labels, ECG) by
        keeping only the moving ultrasound sector, and crops to it.
        Input (N, H, W, 3) uint8, bbox (x0, y0, x1, y1). Output (N, h, w, 3) uint8.
        """
        assert video.ndim == 4, "Video must be a 4D array with shape (num_frames, height, width, channels)."
        x0, y0, x1, y1 = bbox
        return mask_outside_ultrasound_video(video[:, y0:y1, x0:x1], single_mask=True, crop_to_mask=True)

    @staticmethod
    def preprocess_echo_image(image: np.ndarray, bbox_2d: tuple | None, bbox_1d: tuple | None) -> np.ndarray:
        """Burned-in PHI removal for a single-frame image.

        Keeps the full frame but zeroes everything outside the 2D and 1D (M-mode/Doppler) regions.
        Either box may be None; if both are, the whole image is zeroed.
        Input (H, W, 3) uint8, boxes (x0, y0, x1, y1). Output (H, W, 3) uint8.
        """
        assert image.ndim == 3, "Image must be a 3D array with shape (height, width, channels)."
        mask = np.zeros(image.shape[:2], dtype=bool)
        for box in (bbox_2d, bbox_1d):
            if box is not None:
                x0, y0, x1, y1 = box
                mask[y0:y1, x0:x1] = True
        anon_image = np.copy(image)
        anon_image[~mask] = 0
        return anon_image


def _identity(item):
    return item


# ---------------------------------------------------------------------------------------------------
# Models and inference (runs in the GPU process)
# ---------------------------------------------------------------------------------------------------
@dataclass
class Models:
    echofine: Callable[[np.ndarray], np.ndarray]  # (N,224,224,3) uint8 -> (N, n_views) probabilities
    echoprime: Callable[[np.ndarray], np.ndarray]
    echofine_views: list[str]
    echoprime_views: list[str]
    oom_errors: tuple = (torch.OutOfMemoryError,)


def load_models(cfg: dict) -> Models:
    """Load both classifiers once on this process's GPU (visible as device 0).

    TensorFlow is imported here, after CUDA_VISIBLE_DEVICES is set, and never at module level.
    """
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    from src.view_classifier.echofine import ECHOFINE_VIEWS, classify_view_echofine, setup_echofine_view_classifier

    mcfg, chunk = cfg["models"], cfg["preprocess"]["frame_chunk"]
    echofine = setup_echofine_view_classifier(mcfg["echofine_weights"], 0)
    echoprime, echoprime_transforms = setup_echoprime_view_classifier(mcfg["echoprime_weights"], "cuda:0",
                                                                      mcfg["echoprime_compile"])
    return Models(
        echofine=lambda x: classify_view_echofine(echofine, x, batch_size=chunk),
        echoprime=lambda x: classify_view_torch(echoprime, echoprime_transforms, x, batch_size=chunk, device="cuda:0"),
        echofine_views=[ECHOFINE_VIEWS[str(i)] for i in range(len(ECHOFINE_VIEWS))],
        echoprime_views=list(ECHOPRIME_VIEWS),
        oom_errors=(torch.OutOfMemoryError, tf.errors.ResourceExhaustedError),
    )


def summarize(model: str, probits: np.ndarray, views: list[str]) -> dict:
    """Per-view score over frames, score = mean / (1 + std), plus the top-scoring view and its score."""
    score = probits.mean(axis=0) / (1 + probits.std(axis=0))
    out = {f"{model}_View={v}": float(score[i]) for i, v in enumerate(views)}
    best = int(score.argmax())
    out[f"View_{model}"], out[f"ViewScore_{model}"] = views[best], float(score[best])
    return out


def save_media(path: Path, media: np.ndarray, item_type: str, fps: float, codec: str) -> None:
    """Write a PNG (image) or AVI (video) atomically: temp file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")  # keep the extension: cv2/PIL pick the format from it
    if item_type == "image":
        Image.fromarray(media).save(tmp)
    else:
        save_video_cv2(media, str(tmp), fps=fps, codec=codec)
    os.replace(tmp, path)


class ResultWriter:
    """Append-only result parts for one rank: `results/rank=<r>/part-<k>.parquet`, flushed every K rows."""

    def __init__(self, directory: Path, flush_every: int, pipeline_version: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        for tmp in self.dir.glob("*.tmp"):
            tmp.unlink()
        self.next_part = 1 + max((int(p.stem.split("-")[1]) for p in self.dir.glob("part-*.parquet")), default=-1)
        self.flush_every, self.pipeline_version = flush_every, pipeline_version
        self.rows: list[dict] = []

    def add(self, row: dict) -> None:
        self.rows.append({**row, "PipelineVersion": self.pipeline_version, "Timestamp": pd.Timestamp.now(tz="UTC")})
        if len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            atomic_write_parquet(pd.DataFrame(self.rows), self.dir / f"part-{self.next_part:06d}.parquet")
            self.next_part += 1
            self.rows = []


def gpu_handle():
    """NVML handle for this process's GPU, or None if NVML is unavailable (e.g. no driver).

    NVML ignores CUDA_VISIBLE_DEVICES, so the device is looked up by the physical id (index or UUID)
    the parent set for this rank.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        if dev.startswith(("GPU-", "MIG-")):
            return pynvml.nvmlDeviceGetHandleByUUID(dev)
        return pynvml.nvmlDeviceGetHandleByIndex(int(dev))
    except Exception:
        return None


def gpu_usage(handle) -> str:
    """Device-wide GPU utilization and memory (covers both TF and torch), or "" if unavailable."""
    if handle is None:
        return ""
    try:
        import pynvml
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return f"gpu {util}% | mem {mem.used / 1e9:.1f}/{mem.total / 1e9:.1f} GB"
    except Exception:
        return ""


def run_rank(rank: int, files: list[tuple[str, list[dict]]], cfg: dict, models: Models | None) -> Counter:
    """Process one shard of source files with already-loaded models. Returns counts by status.

    With `models=None` (run_echo_preprocessing_and_anonymization.py) the classifiers are skipped: media is saved and the row
    gets status `preprocessed`."""
    pcfg = cfg["preprocess"]
    out_root = Path(cfg["paths"]["output_root"])
    media_root = out_root / "files"
    writer = ResultWriter(out_root / "results" / f"rank={rank}", pcfg["flush_every"], pcfg["pipeline_version"])

    loader_kwargs = dict(batch_size=None, collate_fn=_identity, num_workers=pcfg["num_workers"],
                         pin_memory=torch.cuda.is_available())
    if pcfg["num_workers"] > 0:
        loader_kwargs.update(prefetch_factor=pcfg["prefetch_factor"], persistent_workers=True, in_order=False,
                             multiprocessing_context="spawn")
    loader = DataLoader(AnonymizedEchoDataset(files), **loader_kwargs)

    counts, t0, gpu = Counter(), time.time(), gpu_handle()
    for n_files, item in enumerate(loader, start=1):
        echofine, echofine_error = None, None
        if models is not None and item["echofine_in"] is not None:
            try:
                echofine = summarize("EchoFine", models.echofine(item["echofine_in"]), models.echofine_views)
            except Exception as e:
                echofine_error = type(e).__name__
                if isinstance(e, models.oom_errors):
                    torch.cuda.empty_cache()

        for rec in item["records"]:
            row = {k: rec[k] for k in ("RecordID", "OutputPath", "Status", "ErrorType")}
            if rec["Status"] == "ok":
                if models is None:
                    try:
                        save_media(media_root / rec["OutputPath"], rec["media"], rec["ItemType"], rec["FPS"],
                                   pcfg["video_codec"])
                        row.update(Status="preprocessed")
                    except Exception as e:
                        row.update(Status="error", ErrorType=type(e).__name__)
                elif echofine is None:
                    row.update(Status="error", ErrorType=echofine_error)
                else:
                    try:
                        echoprime = summarize("EchoPrime", models.echoprime(rec["echoprime_in"]), models.echoprime_views)
                        save_media(media_root / rec["OutputPath"], rec["media"], rec["ItemType"], rec["FPS"],
                                   pcfg["video_codec"])
                        row.update(Status="success", **echofine, **echoprime)
                    except Exception as e:
                        row.update(Status="error", ErrorType=type(e).__name__)
                        if isinstance(e, models.oom_errors):
                            torch.cuda.empty_cache()
            writer.add(row)
            counts[row["Status"]] += 1

        if n_files % 100 == 0 or n_files == len(files):
            logger.info("rank {}: {}/{} files ({:.1f} files/s) | {}{}", rank, n_files, len(files),
                        n_files / (time.time() - t0), dict(counts), f" | {u}" if (u := gpu_usage(gpu)) else "")
    writer.flush()
    return counts


def _rank_entry(rank: int, files: list, cfg: dict, log_level: str) -> None:
    """Target of each spawned GPU process. CUDA_VISIBLE_DEVICES was set by the parent before spawning,
    so it was already in place when this process imported torch."""
    setup_logging(Path(cfg["paths"]["output_root"]) / "logs", f"preprocess_rank{rank}", log_level)
    quiet_pydicom()
    logger.info("rank {}: CUDA_VISIBLE_DEVICES={}, {} files", rank, os.environ.get("CUDA_VISIBLE_DEVICES"), len(files))
    models = load_models(cfg)
    logger.info("rank {}: models loaded | {}", rank, gpu_usage(gpu_handle()) or "GPU usage unavailable")
    run_rank(rank, files, cfg, models)


# ---------------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------------
def latest_results(results: pd.DataFrame) -> pd.DataFrame:
    """One row per RecordID: the most recent one."""
    return results.sort_values("Timestamp", kind="stable").drop_duplicates("RecordID", keep="last")


def pending_records(cfg: dict, retry_errors: bool, classify: bool = True) -> pd.DataFrame:
    """Manifest records without a result yet. With `classify`, records only preprocessed (no views) are pending too."""
    manifest = read_parquet_dir(Path(cfg["paths"]["secure_dir"]) / "manifest")
    if not len(manifest):
        return manifest
    results = read_parquet_dir(Path(cfg["paths"]["output_root"]) / "results")
    done = set()
    if len(results):
        latest = latest_results(results)
        if retry_errors:
            latest = latest[latest["Status"] != "error"]
        if classify:
            latest = latest[latest["Status"] != "preprocessed"]
        done = set(latest["RecordID"])
    return manifest[~manifest["RecordID"].isin(done)]


def shard_files(pending: pd.DataFrame, n: int) -> list[list[tuple[str, list[dict]]]]:
    """Group records by source file (decode once), sort files by decoded size (largest first) and deal
    them round-robin across `n` ranks."""
    groups = []
    for source_path, g in pending.groupby("SourcePath", sort=False):
        recs = g[RECORD_FIELDS].astype(object).where(g[RECORD_FIELDS].notna(), None).to_dict("records")
        groups.append((int(g["DecodedBytes"].iloc[0]), source_path, recs))
    groups.sort(key=lambda t: (-t[0], t[1]))
    return [[(src, recs) for _, src, recs in groups[r::n]] for r in range(n)]


def run(cfg: dict, retry_errors: bool, log_level: str) -> bool:
    """Process all pending records. Returns False if any GPU process failed."""
    pending = pending_records(cfg, retry_errors)
    if not len(pending):
        logger.info("Nothing to process")
        return True
    n_gpus = cfg["preprocess"]["num_gpus"]
    shards = shard_files(pending, n_gpus)
    logger.info("{} pending records in {} files, {} GPU processes", len(pending), sum(map(len, shards)), n_gpus)

    # Rank r uses the r-th GPU already visible to this job (respects a scheduler-set CUDA_VISIBLE_DEVICES).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = visible.split(",") if visible else [str(i) for i in range(n_gpus)]
    if len(gpu_ids) < n_gpus:
        raise ValueError(f"num_gpus={n_gpus} but only {len(gpu_ids)} GPUs are visible")

    ctx = mp.get_context("spawn")
    procs = []
    try:
        for rank, files in enumerate(shards):
            if files:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]  # inherited by the child at spawn
                p = ctx.Process(target=_rank_entry, args=(rank, files, cfg, log_level), name=f"rank{rank}")
                p.start()
                procs.append(p)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
    for p in procs:
        p.join()
    failed = [p.name for p in procs if p.exitcode != 0]
    if failed:
        logger.error("GPU processes exited with errors: {} -- rerun to resume", failed)
        return False
    logger.info("All GPU processes finished")
    return True


def study_list(table: pd.DataFrame, crosswalk_path: Path) -> pd.DataFrame:
    """One row per study in the record list: PatientID, StudyID, StudyDateTime (shifted StudyDate +
    StudyTime from the crosswalk; date only if the time is missing, NaT if the date is)."""
    studies = table.reindex(columns=["PatientID", "StudyID", "StudyDateShifted"]).drop_duplicates("StudyID")
    times = pd.read_csv(crosswalk_path, usecols=["AnonStudyID", "StudyTime"], dtype=str, keep_default_na=False)
    times = times.rename(columns={"AnonStudyID": "StudyID"}).drop_duplicates("StudyID")
    studies = studies.merge(times, on="StudyID", how="left")
    date = studies["StudyDateShifted"].fillna("").astype(str).str[:8]
    time = studies["StudyTime"].fillna("").str.strip().str.split(".").str[0].str.ljust(6, "0")
    with_time = pd.to_datetime(date + time, format="%Y%m%d%H%M%S", errors="coerce")
    date_only = pd.to_datetime(date, format="%Y%m%d", errors="coerce")
    studies["StudyDateTime"] = with_time.fillna(date_only)
    return studies[["PatientID", "StudyID", "StudyDateTime"]].sort_values("StudyID").reset_index(drop=True)


def merge(cfg: dict) -> pd.DataFrame:
    """Combine result parts into the final table: latest row per RecordID, joined with the manifest's
    anonymized metadata through an explicit keep-list (no source paths or real identifiers).
    If every manifest record has a non-error result with its media on disk, `results/` is deleted."""
    out_root = Path(cfg["paths"]["output_root"])
    results = read_parquet_dir(out_root / "results")
    manifest = read_parquet_dir(Path(cfg["paths"]["secure_dir"]) / "manifest")
    if not len(results):
        logger.info("No results to merge")
        return pd.DataFrame()

    results = latest_results(results)
    model_cols = [c for c in results.columns if c.startswith(MODEL_COL_PREFIXES)]
    icfg = cfg["index"]
    keep = MANIFEST_KEEP + [f"{k}Shifted" for k in icfg["date_fields"]] + list(icfg["metadata_allowlist"])
    keep = [c for c in dict.fromkeys(keep) if c in manifest.columns]
    table = results[["RecordID"] + RESULT_KEEP + model_cols].merge(manifest[keep], on="RecordID", how="left")
    table = table[keep + RESULT_KEEP + model_cols].sort_values("RecordID").reset_index(drop=True)
    # The anonymized IDs are the only patient/study IDs in the final outputs, so they drop the Anon prefix
    table = table.rename(columns={"AnonPatientID": "PatientID", "AnonStudyID": "StudyID", "OutputPath": "FilePath"})
    atomic_write_parquet(table, out_root / "echo-record-list.parquet")
    studies = study_list(table, Path(cfg["paths"]["secure_dir"]) / "anonymization_crosswalk.csv")
    tmp = out_root / "echo-study-list.csv.tmp"
    studies.to_csv(tmp, index=False)
    os.replace(tmp, out_root / "echo-study-list.csv")

    success = table["Status"].isin(["success", "preprocessed"])
    missing = sum(not (out_root / "files" / p).exists() for p in table.loc[success, "FilePath"])
    errors = int((table["Status"] == "error").sum())
    without_result = int((~manifest["RecordID"].isin(table["RecordID"])).sum())
    logger.info("Final table: {} records -> {}", len(table), out_root / "echo-record-list.parquet")
    logger.info("Study list: {} studies -> {}", len(studies), out_root / "echo-study-list.csv")
    logger.info("QA counts by status: {}", table["Status"].value_counts().to_dict())
    logger.info("QA success records with missing media file: {}", missing)
    logger.info("QA manifest records without a result yet: {}", without_result)
    if missing or errors or without_result:
        logger.info("Merge incomplete: keeping {} so runs can resume", out_root / "results")
    else:
        shutil.rmtree(out_root / "results")
        logger.info("Merge complete: deleted {}", out_root / "results")
    return table


def cli() -> None:
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config")
        p.add_argument("--log-level", default="INFO")
        if name == "run":
            p.add_argument("--retry-errors", action="store_true", help="Reprocess records whose latest status is error")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(Path(cfg["paths"]["output_root"]) / "logs", f"preprocess_{args.command}", args.log_level)
    if args.command == "run":
        if not run(cfg, args.retry_errors, args.log_level):
            raise SystemExit(1)
    else:
        merge(cfg)


if __name__ == "__main__":
    # Re-import under the real module name so the Dataset and process targets pickle as
    # `run_echo_preprocessing_and_anonymization_with_classifier.X` (not `__main__.X`) for the spawned GPU and DataLoader processes.
    import run_echo_preprocessing_and_anonymization_with_classifier
    run_echo_preprocessing_and_anonymization_with_classifier.cli()
