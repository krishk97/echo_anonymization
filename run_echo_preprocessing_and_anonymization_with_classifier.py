"""Script 2 (GPU): preprocesses, classifies and saves every pending manifest record.

Each record is decoded, cropped/masked, classified (EchoFine + EchoPrime) and saved.

Subcommands:
  run    process pending records with one spawned process per GPU
  merge  combine result parts into the final anonymized video and image tables

Restartable: the done set is every RecordID already in a result part (any status);
`--retry-errors` re-queues records whose latest status is `error`. A result row is
written only after both models have run and the media file has been saved
atomically. Records saved by `run_echo_preprocessing_and_anonymization.py` (status
`preprocessed`, no view columns) are not done here: `run` redoes them with
classification.

Model inputs (per source file / record):
  - EchoFine:  the full, unmasked frames (as in the old pipeline), run once per
               source file and shared by all of that file's region records.
  - EchoPrime: the record's preprocessed (cropped + PHI-masked) media.
Only the preprocessed media is written to disk.
"""

import multiprocessing as mp
import os
import shutil
import time
from argparse import ArgumentParser
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.dicom_utils import REGION_DETAIL_TAGS, decode_frames, quiet_pydicom
from src.image_utils import (
    crop_and_scale,
    mask_outside_ultrasound_video,
    save_video_cv2,
)
from src.io_utils import (
    atomic_write_parquet,
    load_config,
    read_parquet_dir,
    setup_logging,
)
from src.view_classifier.echoprime import (
    ECHOPRIME_VIEWS,
    classify_view_torch,
    setup_echoprime_view_classifier,
)

MODEL_RESOLUTION = (224, 224)
# Manifest columns passed to the GPU processes (everything else stays in the manifest)
BBOX_COLUMNS = [
    f"{p}{c}" for p in ("BBox", "BBox2D", "BBox1D") for c in ("X0", "Y0", "X1", "Y1")
]
RECORD_FIELDS = ["RecordID", "OutputPath", "ItemType", "FPS", "Oversize"] + BBOX_COLUMNS
ITEM_TYPES = ("video", "image")


def _region_columns(suffix: str) -> list[str]:
    """Returns the manifest columns of one region ("" video, "2D"/"1D" image)."""
    return [
        f"RegionIndex{suffix}",
        *(f"BBox{suffix}{c}" for c in ("X0", "Y0", "X1", "Y1")),
        f"SpatialFormat{suffix}",
        f"DataType{suffix}",
        *(f"{t}{suffix}" for t in REGION_DETAIL_TAGS),
    ]


# Manifest columns copied into the final tables (besides the config allowlist and
# shifted dates): MANIFEST_KEEP into both, plus VIDEO_KEEP / IMAGE_KEEP. These are
# keep-lists: SourcePath and anything else not listed here never reach the final tables.
MANIFEST_KEEP = [
    "RecordID",
    "AnonPatientID",
    "AnonStudyID",
    "OutputPath",
    "SOPClass",
    "DecodedBytes",
    "Oversize",
    "PatientAgeCapped",
]
VIDEO_KEEP = ["NumFrames", "FPS", "RegionSource"] + _region_columns("")
IMAGE_KEEP = _region_columns("2D") + _region_columns("1D")
RESULT_KEEP = ["Status", "ErrorType", "PipelineVersion", "Timestamp"]
# Prefixes of the view-classification columns written by summarize()
MODEL_COL_PREFIXES = ("EchoFine_", "EchoPrime_", "View_", "ViewScore_")


# --------------------------------------------------------------------------------------
# Data loading (runs in DataLoader worker processes)
# --------------------------------------------------------------------------------------
def _bbox(rec: dict[str, Any], prefix: str) -> tuple[int, int, int, int] | None:
    """Returns the `<prefix>` box (x0, y0, x1, y1) of a manifest record, or None."""
    box = tuple(rec[f"{prefix}{c}"] for c in ("X0", "Y0", "X1", "Y1"))
    return None if box[0] is None else tuple(int(v) for v in box)


def resize_frames(frames: np.ndarray) -> np.ndarray:
    """Resizes (N, H, W, 3) frames to (N, 224, 224, 3) model inputs.

    As in the old pipeline: crop to the aspect ratio, no zoom.

    Args:
        frames: uint8 RGB frames.

    Returns:
        The resized frames.
    """
    return np.stack(
        [crop_and_scale(f, res=MODEL_RESOLUTION, zoom=0.0) for f in frames], axis=0
    )


class AnonymizedEchoDataset(Dataset):
    """Dataset of source files, each with all of its pending records.

    One item is one source file, so a file is decoded only once.

    Attributes:
        files: (source path, pending manifest records) per source file.
    """

    def __init__(self, files: list[tuple[str, list[dict[str, Any]]]]) -> None:
        """Initializes the dataset.

        Args:
            files: (source path, pending manifest records) per source file, as
                returned by `shard_files`.
        """
        self.files = files

    def __len__(self) -> int:
        """Returns the number of source files."""
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Decodes and preprocesses one source file. Never raises.

        Args:
            index: Position in `files`.

        Returns:
            {"echofine_in": (N, 224, 224, 3) array or None, "records": [...]},
            where each record dict has RecordID/OutputPath/ItemType/FPS/Status/
            ErrorType and, when Status is "ok", the preprocessed `media` to save
            and the `echoprime_in` model input. Failures become skip/error
            records.
        """
        source_path, records = self.files[index]
        base = [
            {k: r[k] for k in ("RecordID", "OutputPath", "ItemType", "FPS")}
            for r in records
        ]

        def mark(status: str, err: str | None = None) -> list[dict[str, Any]]:
            return [{**b, "Status": status, "ErrorType": err} for b in base]

        if records[0]["Oversize"]:
            return {"echofine_in": None, "records": mark("skipped_oversize")}
        try:
            frames = decode_frames(source_path)
        except Exception as e:
            return {
                "echofine_in": None,
                "records": mark("skipped_undecodable", type(e).__name__),
            }
        try:
            echofine_in = resize_frames(frames)
        except Exception as e:
            return {"echofine_in": None, "records": mark("error", type(e).__name__)}

        out = []
        for rec, b in zip(records, base, strict=True):
            try:
                if rec["ItemType"] == "video":
                    media = self.preprocess_echo_video(frames, _bbox(rec, "BBox"))
                    echoprime_in = resize_frames(media)
                else:
                    media = self.preprocess_echo_image(
                        frames[0], _bbox(rec, "BBox2D"), _bbox(rec, "BBox1D")
                    )
                    echoprime_in = resize_frames(media[np.newaxis])
                out.append(
                    {
                        **b,
                        "Status": "ok",
                        "ErrorType": None,
                        "media": media,
                        "echoprime_in": echoprime_in,
                    }
                )
            except Exception as e:
                out.append({**b, "Status": "error", "ErrorType": type(e).__name__})
        return {"echofine_in": echofine_in, "records": out}

    @staticmethod
    def preprocess_echo_video(
        video: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Removes burned-in PHI from one video region.

        Crops to the region's bounding box, then zeroes stationary pixels (text,
        labels, ECG) by keeping only the moving ultrasound sector, and crops to it.

        Args:
            video: (N, H, W, 3) uint8 frames.
            bbox: Region box (x0, y0, x1, y1).

        Returns:
            (N, h, w, 3) uint8 masked frames.
        """
        assert video.ndim == 4, (
            "Video must be a 4D array with shape (num_frames, height, width, channels)."
        )
        x0, y0, x1, y1 = bbox
        return mask_outside_ultrasound_video(
            video[:, y0:y1, x0:x1], single_mask=True, crop_to_mask=True
        )

    @staticmethod
    def preprocess_echo_image(
        image: np.ndarray,
        bbox_2d: tuple[int, int, int, int] | None,
        bbox_1d: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        """Removes burned-in PHI from a single-frame image.

        Keeps the full frame but zeroes everything outside the 2D and 1D
        (M-mode/Doppler) regions.

        Args:
            image: (H, W, 3) uint8 frame.
            bbox_2d: 2D region box (x0, y0, x1, y1), or None.
            bbox_1d: 1D region box (x0, y0, x1, y1), or None. If both boxes are
                None, the whole image is zeroed.

        Returns:
            (H, W, 3) uint8 masked image.
        """
        assert image.ndim == 3, (
            "Image must be a 3D array with shape (height, width, channels)."
        )
        mask = np.zeros(image.shape[:2], dtype=bool)
        for box in (bbox_2d, bbox_1d):
            if box is not None:
                x0, y0, x1, y1 = box
                mask[y0:y1, x0:x1] = True
        anon_image = np.copy(image)
        anon_image[~mask] = 0
        return anon_image


def _identity(item: Any) -> Any:
    """Returns `item` unchanged (DataLoader collate function; picklable)."""
    return item


def _init_worker(worker_id: int) -> None:
    """Silences pydicom in a DataLoader worker.

    Spawned workers start fresh, so the rank process's `quiet_pydicom()` call
    doesn't carry over to them.

    Args:
        worker_id: The DataLoader worker index (unused).
    """
    quiet_pydicom()


# --------------------------------------------------------------------------------------
# Models and inference (runs in the GPU process)
# --------------------------------------------------------------------------------------
@dataclass
class Models:
    """Loaded view classifiers.

    Attributes:
        echofine: (N, 224, 224, 3) uint8 -> (N, n_views) probabilities.
        echoprime: Same, for EchoPrime.
        echofine_views: EchoFine view name of each output column.
        echoprime_views: EchoPrime view name of each output column.
        oom_errors: Exception types that mean the GPU ran out of memory.
    """

    echofine: Callable[[np.ndarray], np.ndarray]
    echoprime: Callable[[np.ndarray], np.ndarray]
    echofine_views: list[str]
    echoprime_views: list[str]
    oom_errors: tuple[type[BaseException], ...] = (torch.OutOfMemoryError,)


def load_models(cfg: dict[str, Any]) -> Models:
    """Loads both classifiers once on this process's GPU (visible as device 0).

    TensorFlow is imported here, after CUDA_VISIBLE_DEVICES is set, and never at
    module level.

    Args:
        cfg: The full config.

    Returns:
        The loaded models.
    """
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    from src.view_classifier.echofine import (
        ECHOFINE_VIEWS,
        classify_view_echofine,
        setup_echofine_view_classifier,
    )

    mcfg, chunk = cfg["models"], cfg["preprocess"]["frame_chunk"]
    echofine = setup_echofine_view_classifier(mcfg["echofine_weights"], 0)
    echoprime, echoprime_transforms = setup_echoprime_view_classifier(
        mcfg["echoprime_weights"], "cuda:0", mcfg["echoprime_compile"]
    )
    return Models(
        echofine=lambda x: classify_view_echofine(echofine, x, batch_size=chunk),
        echoprime=lambda x: classify_view_torch(
            echoprime, echoprime_transforms, x, batch_size=chunk, device="cuda:0"
        ),
        echofine_views=[ECHOFINE_VIEWS[str(i)] for i in range(len(ECHOFINE_VIEWS))],
        echoprime_views=list(ECHOPRIME_VIEWS),
        oom_errors=(torch.OutOfMemoryError, tf.errors.ResourceExhaustedError),
    )


def summarize(model: str, probits: np.ndarray, views: list[str]) -> dict[str, Any]:
    """Reduces per-frame probabilities to one score per view.

    Args:
        model: Model name used in the column names ("EchoFine" or "EchoPrime").
        probits: (N, n_views) per-frame probabilities.
        views: View name of each column of `probits`.

    Returns:
        `<model>_View=<view>` scores (score = mean / (1 + std) over frames), plus
        `View_<model>` (top-scoring view) and `ViewScore_<model>` (its score).
    """
    score = probits.mean(axis=0) / (1 + probits.std(axis=0))
    out = {f"{model}_View={v}": float(score[i]) for i, v in enumerate(views)}
    best = int(score.argmax())
    out[f"View_{model}"], out[f"ViewScore_{model}"] = views[best], float(score[best])
    return out


def save_media(
    path: Path, media: np.ndarray, item_type: str, fps: float, codec: str
) -> None:
    """Writes a PNG (image) or AVI (video) atomically.

    The media goes to a temp file in the same directory, which is then renamed.

    Args:
        path: Destination path.
        media: (H, W, 3) image or (N, H, W, 3) video, uint8 RGB.
        item_type: "image" or "video".
        fps: Video frame rate.
        codec: FourCC codec for videos.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the extension: cv2/PIL pick the format from it.
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if item_type == "image":
        Image.fromarray(media).save(tmp)
    else:
        save_video_cv2(media, str(tmp), fps=fps, codec=codec)
    os.replace(tmp, path)


class ResultWriter:
    """Append-only result parts for one rank.

    Rows are buffered and written as `results/rank=<r>/part-<k>.parquet` every
    `flush_every` rows.

    Attributes:
        dir: The rank's results directory.
        next_part: Number of the next part file.
        flush_every: Rows per part file.
        pipeline_version: Value of the PipelineVersion column.
        rows: Buffered rows not yet written.
    """

    def __init__(
        self, directory: Path, flush_every: int, pipeline_version: str
    ) -> None:
        """Opens the rank's results directory, removing leftover temp files.

        Args:
            directory: The rank's results directory; created if missing.
            flush_every: Rows per part file.
            pipeline_version: Value of the PipelineVersion column.
        """
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        for tmp in self.dir.glob("*.tmp"):
            tmp.unlink()
        self.next_part = 1 + max(
            (int(p.stem.split("-")[1]) for p in self.dir.glob("part-*.parquet")),
            default=-1,
        )
        self.flush_every, self.pipeline_version = flush_every, pipeline_version
        self.rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        """Buffers a row (adding PipelineVersion and Timestamp), flushing if full."""
        self.rows.append(
            {
                **row,
                "PipelineVersion": self.pipeline_version,
                "Timestamp": pd.Timestamp.now(tz="UTC"),
            }
        )
        if len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Writes buffered rows as the next part file, if there are any."""
        if self.rows:
            atomic_write_parquet(
                pd.DataFrame(self.rows), self.dir / f"part-{self.next_part:06d}.parquet"
            )
            self.next_part += 1
            self.rows = []


def gpu_handle() -> Any:
    """Returns the NVML handle for this process's GPU.

    NVML ignores CUDA_VISIBLE_DEVICES, so the device is looked up by the physical
    id (index or UUID) the parent set for this rank.

    Returns:
        The handle, or None if NVML is unavailable (e.g. no driver).
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


def gpu_usage(handle: Any) -> str:
    """Describes device-wide GPU utilization and memory (covers both TF and torch).

    Args:
        handle: Handle from `gpu_handle`, or None.

    Returns:
        A short usage string, or "" if unavailable.
    """
    if handle is None:
        return ""
    try:
        import pynvml

        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return f"gpu {util}% | mem {mem.used / 1e9:.1f}/{mem.total / 1e9:.1f} GB"
    except Exception:
        return ""


def run_rank(
    rank: int,
    files: list[tuple[str, list[dict[str, Any]]]],
    cfg: dict[str, Any],
    models: Models | None,
) -> Counter:
    """Processes one shard of source files with already-loaded models.

    With `models=None` (run_echo_preprocessing_and_anonymization.py) the
    classifiers are skipped: media is saved and the row gets status
    `preprocessed`.

    Args:
        rank: This process's rank; selects `results/rank=<rank>/`.
        files: The shard, from `shard_files`.
        cfg: The full config.
        models: Loaded models, or None to skip classification.

    Returns:
        Number of records by result status.
    """
    pcfg = cfg["preprocess"]
    out_root = Path(cfg["paths"]["output_root"])
    media_root = out_root / "files"
    writer = ResultWriter(
        out_root / "results" / f"rank={rank}",
        pcfg["flush_every"],
        pcfg["pipeline_version"],
    )

    loader_kwargs = dict(
        batch_size=None,
        collate_fn=_identity,
        num_workers=pcfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    if pcfg["num_workers"] > 0:
        loader_kwargs.update(
            prefetch_factor=pcfg["prefetch_factor"],
            persistent_workers=True,
            in_order=False,
            multiprocessing_context="spawn",
            worker_init_fn=_init_worker,
        )
    loader = DataLoader(AnonymizedEchoDataset(files), **loader_kwargs)

    counts, t0, gpu = Counter(), time.time(), gpu_handle()
    for n_files, item in enumerate(loader, start=1):
        echofine, echofine_error = None, None
        if models is not None and item["echofine_in"] is not None:
            try:
                echofine = summarize(
                    "EchoFine",
                    models.echofine(item["echofine_in"]),
                    models.echofine_views,
                )
            except Exception as e:
                echofine_error = type(e).__name__
                if isinstance(e, models.oom_errors):
                    torch.cuda.empty_cache()

        for rec in item["records"]:
            row = {k: rec[k] for k in ("RecordID", "OutputPath", "Status", "ErrorType")}
            if rec["Status"] == "ok":
                if models is None:
                    try:
                        save_media(
                            media_root / rec["OutputPath"],
                            rec["media"],
                            rec["ItemType"],
                            rec["FPS"],
                            pcfg["video_codec"],
                        )
                        row.update(Status="preprocessed")
                    except Exception as e:
                        row.update(Status="error", ErrorType=type(e).__name__)
                elif echofine is None:
                    row.update(Status="error", ErrorType=echofine_error)
                else:
                    try:
                        echoprime = summarize(
                            "EchoPrime",
                            models.echoprime(rec["echoprime_in"]),
                            models.echoprime_views,
                        )
                        save_media(
                            media_root / rec["OutputPath"],
                            rec["media"],
                            rec["ItemType"],
                            rec["FPS"],
                            pcfg["video_codec"],
                        )
                        row.update(Status="success", **echofine, **echoprime)
                    except Exception as e:
                        row.update(Status="error", ErrorType=type(e).__name__)
                        if isinstance(e, models.oom_errors):
                            torch.cuda.empty_cache()
            writer.add(row)
            counts[row["Status"]] += 1

        if n_files % 100 == 0 or n_files == len(files):
            logger.info(
                "rank {}: {}/{} files ({:.1f} files/s) | {}{}",
                rank,
                n_files,
                len(files),
                n_files / (time.time() - t0),
                dict(counts),
                f" | {u}" if (u := gpu_usage(gpu)) else "",
            )
    writer.flush()
    return counts


def _rank_entry(
    rank: int,
    files: list[tuple[str, list[dict[str, Any]]]],
    cfg: dict[str, Any],
    log_level: str,
) -> None:
    """Runs one spawned GPU process: sets up logging, loads models, runs its shard.

    CUDA_VISIBLE_DEVICES was set by the parent before spawning, so it was already
    in place when this process imported torch.
    """
    setup_logging(
        Path(cfg["paths"]["output_root"]) / "logs", f"preprocess_rank{rank}", log_level
    )
    quiet_pydicom()
    logger.info(
        "rank {}: CUDA_VISIBLE_DEVICES={}, {} files",
        rank,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        len(files),
    )
    models = load_models(cfg)
    logger.info(
        "rank {}: models loaded | {}",
        rank,
        gpu_usage(gpu_handle()) or "GPU usage unavailable",
    )
    run_rank(rank, files, cfg, models)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def latest_results(results: pd.DataFrame) -> pd.DataFrame:
    """Keeps the most recent row per RecordID."""
    return results.sort_values("Timestamp", kind="stable").drop_duplicates(
        "RecordID", keep="last"
    )


def read_manifests(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Reads the video and image manifests.

    Args:
        cfg: The full config.

    Returns:
        {"video": video manifest, "image": image manifest}; either may be empty.
    """
    manifest_dir = Path(cfg["paths"]["secure_dir"]) / "manifest"
    return {t: read_parquet_dir(manifest_dir / t) for t in ITEM_TYPES}


def pending_records(
    cfg: dict[str, Any], retry_errors: bool, classify: bool = True
) -> pd.DataFrame:
    """Returns the manifest records without a result yet.

    A record whose latest result was written with a different
    `preprocess.pipeline_version` than the config's is pending again.

    Args:
        cfg: The full config.
        retry_errors: Also return records whose latest status is `error`.
        classify: Also return records that were only preprocessed (no views).

    Returns:
        The pending video and image manifest rows (columns of the other type are
        NaN).
    """
    manifest = pd.concat(read_manifests(cfg).values(), ignore_index=True)
    if not len(manifest):
        return manifest
    results = read_parquet_dir(Path(cfg["paths"]["output_root"]) / "results")
    done = set()
    if len(results):
        latest = latest_results(results)
        version = cfg["preprocess"]["pipeline_version"]
        latest = latest[latest["PipelineVersion"] == version]
        if retry_errors:
            latest = latest[latest["Status"] != "error"]
        if classify:
            latest = latest[latest["Status"] != "preprocessed"]
        done = set(latest["RecordID"])
    return manifest[~manifest["RecordID"].isin(done)]


def shard_files(
    pending: pd.DataFrame, n: int
) -> list[list[tuple[str, list[dict[str, Any]]]]]:
    """Groups records by source file and deals the files across ranks.

    Grouping means each file is decoded once. Files are sorted by decoded size
    (largest first) and dealt round-robin.

    Args:
        pending: Pending manifest rows, from `pending_records`.
        n: Number of ranks.

    Returns:
        One shard per rank: (source path, record dicts) per file.
    """
    groups = []
    for source_path, g in pending.groupby("SourcePath", sort=False):
        # Videos have no image boxes and vice versa; a column may be missing entirely.
        fields = g.reindex(columns=RECORD_FIELDS)
        recs = fields.astype(object).where(fields.notna(), None).to_dict("records")
        groups.append((int(g["DecodedBytes"].iloc[0]), source_path, recs))
    groups.sort(key=lambda t: (-t[0], t[1]))
    return [[(src, recs) for _, src, recs in groups[r::n]] for r in range(n)]


def run(cfg: dict[str, Any], retry_errors: bool, log_level: str) -> bool:
    """Processes all pending records with one spawned process per GPU.

    Args:
        cfg: The full config.
        retry_errors: Re-queue records whose latest status is `error`.
        log_level: Log level for the spawned processes.

    Returns:
        False if any GPU process failed, else True.

    Raises:
        ValueError: Fewer GPUs are visible than `preprocess.num_gpus`.
    """
    pending = pending_records(cfg, retry_errors)
    if not len(pending):
        logger.info("Nothing to process")
        return True
    n_gpus = cfg["preprocess"]["num_gpus"]
    shards = shard_files(pending, n_gpus)
    logger.info(
        "{} pending records in {} files, {} GPU processes",
        len(pending),
        sum(map(len, shards)),
        n_gpus,
    )

    # Rank r uses the r-th GPU already visible to this job (respects a scheduler-set
    # CUDA_VISIBLE_DEVICES).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = visible.split(",") if visible else [str(i) for i in range(n_gpus)]
    if len(gpu_ids) < n_gpus:
        raise ValueError(f"num_gpus={n_gpus} but only {len(gpu_ids)} GPUs are visible")

    ctx = mp.get_context("spawn")
    procs = []
    try:
        for rank, files in enumerate(shards):
            if files:
                # Inherited by the child at spawn.
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
                p = ctx.Process(
                    target=_rank_entry,
                    args=(rank, files, cfg, log_level),
                    name=f"rank{rank}",
                )
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
    """Builds the study list from the final tables.

    Args:
        table: The final video and image tables combined (with PatientID/StudyID
            already renamed).
        crosswalk_path: The crosswalk CSV; only AnonStudyID and StudyTime are read.

    Returns:
        One row per study: PatientID, StudyID, StudyDateTime (shifted StudyDate +
        StudyTime from the crosswalk; date only if the time is missing, NaT if the
        date is).
    """
    studies = table.reindex(
        columns=["PatientID", "StudyID", "StudyDateShifted"]
    ).drop_duplicates("StudyID")
    times = pd.read_csv(
        crosswalk_path,
        usecols=["AnonStudyID", "StudyTime"],
        dtype=str,
        keep_default_na=False,
    )
    times = times.rename(columns={"AnonStudyID": "StudyID"}).drop_duplicates("StudyID")
    studies = studies.merge(times, on="StudyID", how="left")
    date = studies["StudyDateShifted"].fillna("").astype(str).str[:8]
    time = (
        studies["StudyTime"]
        .fillna("")
        .str.strip()
        .str.split(".")
        .str[0]
        .str.ljust(6, "0")
    )
    with_time = pd.to_datetime(date + time, format="%Y%m%d%H%M%S", errors="coerce")
    date_only = pd.to_datetime(date, format="%Y%m%d", errors="coerce")
    studies["StudyDateTime"] = with_time.fillna(date_only)
    return (
        studies[["PatientID", "StudyID", "StudyDateTime"]]
        .sort_values("StudyID")
        .reset_index(drop=True)
    )


def merge(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combines result parts into the final video and image tables and study list.

    Takes the latest row per RecordID, joined with the video or image manifest's
    anonymized metadata through an explicit keep-list (no source paths or real
    identifiers). If every manifest record has a non-error result from the current
    `pipeline_version` with its media on disk, `results/` is deleted.

    Args:
        cfg: The full config.

    Returns:
        (video table, image table); both empty if there are no results.
    """
    out_root = Path(cfg["paths"]["output_root"])
    results = read_parquet_dir(out_root / "results")
    manifests = read_manifests(cfg)
    if not len(results):
        logger.info("No results to merge")
        return pd.DataFrame(), pd.DataFrame()

    results = latest_results(results)
    model_cols = [c for c in results.columns if c.startswith(MODEL_COL_PREFIXES)]
    icfg = cfg["index"]
    common = (
        MANIFEST_KEEP
        + [f"{k}Shifted" for k in icfg["date_fields"]]
        + list(icfg["metadata_allowlist"])
    )
    tables = {}
    for item_type, type_keep in (("video", VIDEO_KEEP), ("image", IMAGE_KEEP)):
        manifest = manifests[item_type]
        keep = [c for c in dict.fromkeys(common + type_keep) if c in manifest.columns]
        table = pd.DataFrame()
        if keep:  # Empty if this run has no records of this type
            table = results[["RecordID"] + RESULT_KEEP + model_cols].merge(
                manifest[keep], on="RecordID", how="inner"
            )
            table = (
                table[keep + RESULT_KEEP + model_cols]
                .sort_values("RecordID")
                .reset_index(drop=True)
            )
            # The anonymized IDs are the only patient/study IDs in the final
            # outputs, so they drop the Anon prefix
            table = table.rename(
                columns={
                    "AnonPatientID": "PatientID",
                    "AnonStudyID": "StudyID",
                    "OutputPath": "FilePath",
                }
            )
        tables[item_type] = table
        path = out_root / f"echo-{item_type}-list.parquet"
        atomic_write_parquet(tables[item_type], path)
        logger.info("Final {} table: {} records -> {}", item_type, len(table), path)

    both = pd.concat(tables.values(), ignore_index=True)
    studies = study_list(
        both, Path(cfg["paths"]["secure_dir"]) / "anonymization_crosswalk.csv"
    )
    tmp = out_root / "echo-study-list.csv.tmp"
    studies.to_csv(tmp, index=False)
    os.replace(tmp, out_root / "echo-study-list.csv")

    success = both["Status"].isin(["success", "preprocessed"])
    missing = sum(
        not (out_root / "files" / p).exists() for p in both.loc[success, "FilePath"]
    )
    errors = int((both["Status"] == "error").sum())
    manifest_ids = pd.concat(manifests.values())["RecordID"]
    without_result = int((~manifest_ids.isin(both["RecordID"])).sum())
    version = cfg["preprocess"]["pipeline_version"]
    stale = int((both["PipelineVersion"] != version).sum())
    logger.info(
        "Study list: {} studies -> {}", len(studies), out_root / "echo-study-list.csv"
    )
    logger.info("QA counts by status: {}", both["Status"].value_counts().to_dict())
    logger.info("QA success records with missing media file: {}", missing)
    logger.info("QA manifest records without a result yet: {}", without_result)
    logger.info("QA records from another pipeline_version: {}", stale)
    if missing or errors or without_result or stale:
        logger.info(
            "Merge incomplete: keeping {} so runs can resume", out_root / "results"
        )
    else:
        shutil.rmtree(out_root / "results")
        logger.info("Merge complete: deleted {}", out_root / "results")
    return tables["video"], tables["image"]


def cli() -> None:
    """Parses the command line and runs `run` or `merge`."""
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "merge"):
        p = sub.add_parser(name)
        p.add_argument(
            "--config", default="configs/config.yaml", help="Path to the YAML config"
        )
        p.add_argument("--log-level", default="INFO")
        if name == "run":
            p.add_argument(
                "--retry-errors",
                action="store_true",
                help="Reprocess records whose latest status is error",
            )
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(
        Path(cfg["paths"]["output_root"]) / "logs",
        f"preprocess_{args.command}",
        args.log_level,
    )
    if args.command == "run":
        if not run(cfg, args.retry_errors, args.log_level):
            raise SystemExit(1)
    else:
        merge(cfg)


if __name__ == "__main__":
    # Re-import under the real module name so the Dataset and process targets pickle
    # as `run_echo_preprocessing_and_anonymization_with_classifier.X` (not
    # `__main__.X`) for the spawned GPU and DataLoader processes.
    import run_echo_preprocessing_and_anonymization_with_classifier

    run_echo_preprocessing_and_anonymization_with_classifier.cli()
