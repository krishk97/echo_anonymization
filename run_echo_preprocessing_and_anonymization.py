"""Script 2 without the view classifiers (CPU only): decode, crop/mask and save every pending manifest record.

For when no GPUs are available. Uses the same Dataset, sharding and result parts as
`run_echo_preprocessing_and_anonymization_with_classifier.py run`, with `preprocess.num_cpu_procs` processes instead of one per GPU.
Rows get status `preprocessed` (no view columns). A later `run_echo_preprocessing_and_anonymization_with_classifier.py run` treats them
as pending and redoes them with classification; `run_echo_preprocessing_and_anonymization_with_classifier.py merge` builds the final
table as usual. Don't run this and `run_echo_preprocessing_and_anonymization_with_classifier.py run` at the same time.
"""
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger

from run_echo_preprocessing_and_anonymization_with_classifier import pending_records, run_rank, shard_files
from src.dicom_utils import quiet_pydicom
from src.io_utils import load_config, setup_logging


def _rank_entry(rank: int, files: list, cfg: dict, log_level: str) -> None:
    """Target of each spawned CPU process."""
    setup_logging(Path(cfg["paths"]["output_root"]) / "logs", f"run_echo_preprocessing_and_anonymization_rank{rank}", log_level)
    quiet_pydicom()
    logger.info("rank {}: {} files", rank, len(files))
    run_rank(rank, files, cfg, None)


def run(cfg: dict, retry_errors: bool, log_level: str) -> bool:
    """Preprocess all pending records. Returns False if any process failed."""
    pending = pending_records(cfg, retry_errors, classify=False)
    if not len(pending):
        logger.info("Nothing to process")
        return True
    n_procs = cfg["preprocess"]["num_cpu_procs"]
    shards = shard_files(pending, n_procs)
    logger.info("{} pending records in {} files, {} CPU processes", len(pending), sum(map(len, shards)), n_procs)

    ctx = mp.get_context("spawn")
    procs = []
    for rank, files in enumerate(shards):
        if files:
            p = ctx.Process(target=_rank_entry, args=(rank, files, cfg, log_level), name=f"rank{rank}")
            p.start()
            procs.append(p)
    for p in procs:
        p.join()
    failed = [p.name for p in procs if p.exitcode != 0]
    if failed:
        logger.error("Processes exited with errors: {} -- rerun to resume", failed)
        return False
    logger.info("All processes finished")
    return True


def cli() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--retry-errors", action="store_true", help="Reprocess records whose latest status is error")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(Path(cfg["paths"]["output_root"]) / "logs", "run_echo_preprocessing_and_anonymization", args.log_level)
    if not run(cfg, args.retry_errors, args.log_level):
        raise SystemExit(1)


if __name__ == "__main__":
    # Re-import under the real module name so the process target pickles as `run_echo_preprocessing_and_anonymization._rank_entry`
    import run_echo_preprocessing_and_anonymization
    run_echo_preprocessing_and_anonymization.cli()
