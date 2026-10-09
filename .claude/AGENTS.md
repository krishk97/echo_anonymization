# AGENTS.md — repository context for coding agents

Context for AI agents working in this repo. Behavioural rules for agents are in `.claude/CLAUDE.md`. User-facing docs (config keys, run commands, output layout) are in `README.md`; this file doesn't repeat them.

## What this repo is

A two-stage pipeline that de-identifies a large (>1M file) echocardiogram DICOM dataset and classifies each record's view.

1. **`create_anonymization_plan.py`** (Script 1, CPU) reads headers only, without decoding pixels.
   - Keeps only the SOP classes `Ultrasound Image Storage` and `Ultrasound Multi-frame Image Storage`.
   - Extends an append-only study-level **crosswalk** CSV.
   - Finds ultrasound regions.
   - Writes an item-level **work manifest** (Parquet parts) plus a per-file **status log**.
2. **`run_echo_preprocessing_and_anonymization_with_classifier.py`** (Script 2, GPU) has two subcommands:
   - `run` decodes each pending manifest record, crops and masks it, runs **EchoFine** (TF/Keras, 47 views) and **EchoPrime** (PyTorch, 11 views), saves PNG/AVI media under `output_root/file/`, and appends result parts. It spawns one process per GPU.
   - `merge` builds the final anonymized `echo-record-list.parquet` and `echo-study-list.csv` (`PatientID`, `StudyID`, `StudyDateTime`). In both, the anonymized `AnonPatientID`/`AnonStudyID` are renamed to `PatientID`/`StudyID`; in the record list `OutputPath` is also renamed to `FilePath`. For the study list it reads only the `AnonStudyID` and `StudyTime` columns of the crosswalk; `StudyDateTime` = shifted StudyDate + unshifted StudyTime.

3. **`run_echo_preprocessing_and_anonymization.py`** (CPU) is Script 2's `run` without the view classifiers, for when no GPU is available. It reuses `pending_records(..., classify=False)`, `shard_files` and `run_rank(..., models=None)`, and spawns `preprocess.num_cpu_procs` processes. It saves the same media, and its result rows get status `preprocessed` with no view columns. The final table is still built with `run_echo_preprocessing_and_anonymization_with_classifier.py merge`.

All settings live in `configs/config.yaml`, which both scripts share. The user is meant to edit only this file.

## Hard rules

- **Never open, print or log real DICOM headers or pixel data on this server.** Develop and test only against `tests/synthetic.py` fixtures or pydicom test data. Do not run either script against real paths.
- **Logs and status rows record the exception *type* only**, never messages, tag values or file paths. pydicom's logger and warnings are silenced via `quiet_pydicom()` for the same reason.
- **The crosswalk is append-only.** Existing IDs must never change, and counters resume from the highest existing ID.
  - IDs are sequential, with no hashing and no secret key: `P000000` per `PatientID`, `S000000` per `StudyInstanceUID`.
  - Only Script 1's main process writes the crosswalk. Workers just return the keys they read.
- **Metadata leaves the secure area only through allowlists:**
  - `index.metadata_allowlist` in the config;
  - `MANIFEST_KEEP` / `RESULT_KEEP` in `run_echo_preprocessing_and_anonymization_with_classifier.py`, which `merge` uses as a keep-list rather than a drop-list.
  - `SourcePath` and the crosswalk fields must never reach `output_root`.
- **The manifest and status parts contain source paths, so they are PHI** and live in `paths.secure_dir`. Anything under `paths.output_root` must be anonymized.

## Layout

```
create_anonymization_plan.py                                 Script 1. Also holds the output-naming functions (assign_stem, record_id_for, stem_of).
                                                             find_files_by_extension is the user's helper; keep it.
run_echo_preprocessing_and_anonymization_with_classifier.py  Script 2: AnonymizedEchoDataset, Models/load_models, run_rank, ResultWriter,
                                                             pending_records, shard_files, run, merge, cli
run_echo_preprocessing_and_anonymization.py                  CPU-only Script 2 without classifiers: _rank_entry, run, cli (reuses Script 2's functions)
src/anon_crosswalk.py                                        AnonCrosswalk class: load + truncated-last-line recovery, assign(), append+fsync
src/dicom_utils.py                                           read_header (read_partial, stop at pixel data; reports has_pixels), sop_class_name,
                                                             to_primitive, compute_fps, shift_date, cap_age, extract_header (worker entry),
                                                             Region + find_us_regions + select_video_regions/select_image_regions,
                                                             decode_frames (frame-level iter_pixels -> (N,H,W,3) uint8 RGB)
src/image_utils.py                                           (pre-existing, user's) save_video_cv2, mask_outside_ultrasound_video, crop_and_scale
src/io_utils.py                                              load_config, atomic_write_parquet (tmp + fsync + os.replace), read_parquet_dir, setup_logging (Loguru sinks)
src/view_classifier/                                         (pre-existing, user's) echofine.py (imports TF at module level), echoprime.py
configs/config.yaml                                          the only config
scripts/*.sh                                                 uv run wrappers for the stages, incl. run_preprocess_only.sh
slurm/                                                       sbatch examples (in README "On a Slurm cluster"): 01_create_anonymization_plan, 02_run_with_classifier
                                                             (GPU run + merge), 02_run_preprocess_only (CPU + merge), submit_pipeline.sh
                                                             (chains 01 -> 02 GPU via afterok). Submit from repo root (cd $SLURM_SUBMIT_DIR);
                                                             stdout goes to slurm/logs/. --gres/--cpus-per-task must match config.yaml by hand.
```

### Internal-only (local, not part of the repo)

The user keeps these on disk for their own use, and they are deliberately **not** part of the published repo.
They are gitignored or deleted. **Don't mention them in `README.md`** or any other user-facing docs, and don't
add them to version control.

- `tests/`: `conftest.py` (fixtures `dicom_tree`, `make_cfg`), `synthetic.py` (fake DICOM generator) and
  `test_index.py` (Script 1), `test_preprocess.py` / `test_merge.py` (Script 2, stub models). It's gitignored.
- `example_old_code/`: the user's earlier `anonymize_echos.py` / `preprocess_echos.py`, which this pipeline
  iterates on. It has been removed from the working tree and may come back locally.

## Data flow and on-disk contracts

**Script 1**
- Files are found, sorted by path, and split into fixed-size chunks that a `mp.Pool` processes. Each chunk is committed in this order:
  1. `crosswalk.assign()`: new IDs, in sorted (PatientID, StudyUID) order, are appended and fsynced.
  2. `manifest/chunk-NNNNNN.parquet` is written atomically.
  3. `status/chunk-NNNNNN.parquet` is written atomically. This is the **commit marker**.
- On startup:
  - leftover `*.tmp` files are removed;
  - manifest parts without a matching status part are deleted;
  - done = the source paths in all status parts;
  - the naming registry = the stems of all manifest `RecordID`s.
- Part numbers form one global sequence across runs. Don't put timestamps in part names: a same-second restart once overwrote a committed chunk.
- Status values: `included`, `excluded_sop_class` (detail = SOP name), `excluded_no_pixel_data`, `excluded_missing_patient_id`, `excluded_missing_study_uid`, `excluded_no_video_region`, `error` (detail = exception type).

**Script 2**
- Done = every `RecordID` in `output_root/results/rank=*/part-*.parquet`, whatever its status. With `--retry-errors`, items whose latest row is `error` are re-queued. Items whose latest row is `preprocessed` are pending for `run_echo_preprocessing_and_anonymization_with_classifier.py run` (`classify=True`, the default) but done for `run_echo_preprocessing_and_anonymization.py` (`classify=False`).
- Pending records are grouped **by source file** and sorted by `DecodedBytes` (largest first), then dealt round-robin to the ranks.
- The parent sets `CUDA_VISIBLE_DEVICES` before each `spawn` and maps rank r to the r-th GPU already visible to the job. TF is imported only inside `load_models` and enables memory growth.
- DataLoader: `batch_size=None`, identity collate, `multiprocessing_context="spawn"`, `in_order=False`, `persistent_workers=True`.
- A result row is added only after both models have run **and** the media file has been saved atomically. The temp name keeps the extension (`x.tmp.avi`) because cv2/PIL choose the format from it.
- Result statuses: `success`, `preprocessed` (run_echo_preprocessing_and_anonymization.py only), `skipped_oversize`, `skipped_undecodable`, `error`. With `models=None`, `run_rank` skips both models and writes `preprocessed` once the media is saved. The `merge` QA check for missing media covers `success` and `preprocessed`.
- `run_echo_preprocessing_and_anonymization.py` and the GPU `run` share `results/rank=<r>/`. Running them at the same time could clash on part numbers.
- `__main__` re-imports the module as `run_echo_preprocessing_and_anonymization_with_classifier` (`run_echo_preprocessing_and_anonymization.py` does the same with its own name), so the Dataset and process targets pickle under the real module name for spawned children. Keep that.
- `run` returns False and the CLI exits 1 if any rank process fails.
- GPU usage logging uses `pynvml` (from the `nvidia-ml-py` dependency). `gpu_handle()` looks the device up by the physical id in `CUDA_VISIBLE_DEVICES` (index or `GPU-`/`MIG-` UUID), because NVML ignores that variable. `gpu_usage()` reports device-wide utilization and memory, so TF and torch are both counted. Each rank logs it once after the models load and on every progress line. Without a driver, both helpers return None/"" and the log line stays unchanged.

## Decisions confirmed by the user (don't change without asking)

- **Output naming** (`<AnonPatientID>/<AnonStudyID>/<name>`, first claimer wins, a name never changes):
  - The base is the InstanceNumber.
  - Every record of a multi-region video gets `_r<region_index>`.
  - A later clash in the same study gets `_s<SeriesNumber>`, or `_sna` if SeriesNumber is missing; a further clash adds `-2`, `-3`, ….
  - A missing InstanceNumber gives `na<k>` (next free k).
  - The registry stores stems, i.e. the RecordID without `_r<i>`.
- **Video regions:** keep regions with RegionSpatialFormat = 2D and RegionDataType ∈ {unknown/0, Tissue, Color Flow}.
  - Drop a region whose box lies fully inside another kept region.
  - No region sequence → one full-frame record (`RegionSource="full_frame"`).
  - No qualifying region → `excluded_no_video_region`.
- **Images:** one record each.
  - `BBox2D*` = the largest 2D region; `BBox1D*` = the largest M-mode/Spectral region. Either may be null.
  - All regions go into `RegionsJSON`.
  - No region sequence → `BBox2D*` is the full frame.
- **Boxes are half-open** `(x0, y0, x1, y1)`, ready for numpy slicing. DICOM's MaxX1/MaxY1 are inclusive, so 1 is added.
- **Missing PatientID or StudyInstanceUID** → the file is excluded with a status. It is never grouped with other files.
- **Date shift:** one random integer offset per patient, in [36159, 36525] days (+99–100 y). It is drawn via `secrets.SystemRandom` and reused for all of that patient's studies. Ages of 90 or more become `"90+"`.
- **Model inputs:**
  - EchoFine gets the **full, unmasked frames** (`crop_and_scale(..., zoom=0)` to 224). It runs once per source file, and all region records of that file share its output.
  - EchoPrime gets the record's masked media, resized to 224.
- **View-classification columns (user decision, 2026-10-08):** `summarize()` stores one score per view, `score = mean / (1 + std)` of the per-frame probabilities, as `EchoFine_View=<view>` / `EchoPrime_View=<view>`, plus `View_<Model>` (argmax of the score) and `ViewScore_<Model>`. Raw mean/std are not saved. `merge` picks these columns up via `MODEL_COL_PREFIXES`. Results written earlier used `echofine_<view>`, `_std` and `echofine_view`.
- **Only the masked media is written to disk**, at native (cropped) resolution:
  - Videos are cropped to the region box, then `mask_outside_ultrasound_video(single_mask=True, crop_to_mask=True)`.
  - Images keep the full frame, with everything outside `BBox2D ∪ BBox1D` zeroed.
- **Environment:** uv with `python-preference = "only-managed"`, Python 3.11, `.venv`. torch/torchvision come from the cu128 index; the dependencies include `tensorflow[and-cuda]`, `tf_keras` and `loguru`.

## Code style

- Write and review Python against the `python-style-guide` skill (`.claude/skills/python-style-guide/`, Google style). Load it before writing Python.
- Apply it to new and changed code only. The existing code was written before the guide, and much of it doesn't follow it: about 250 lines are over 88 characters, most functions have one-line docstrings or none instead of Google-style `Args:`/`Returns:`, and it imports names from app modules (`from src.io_utils import load_config, ...`). Don't reformat or re-docstring untouched code unless the user asks.
- **Logging uses Loguru** (user decision, 2026-10-08), as the guide says: `from loguru import logger`, brace-style lazy args (`logger.info("{} files", n)`), no `logging.getLogger`. `setup_logging()` in `src/io_utils.py` calls `logger.remove()`, then adds a stderr sink and a `<log_dir>/<name>_<timestamp>.log` sink. Each spawned rank process calls it again. The `{name}` field is the module name, so Script 1 logs as `__main__`. Exceptions to note:
  - Log the exception type only (see Hard rules), never `{e}` as the guide's examples do.
  - `quiet_pydicom()` still uses stdlib `logging`, because pydicom logs through it. Other third-party stdlib loggers (TF, torch) are no longer routed into the log files; they fall back to stdlib's default stderr handler (WARNING and above).
  - pytest's `caplog` doesn't see Loguru. A test that checks log text adds a sink on `caplog.handler` (see `tests/test_merge.py: test_merge_counts_missing_media`).
- **Keep the `from src.<module> import <name>` style** for app imports, to stay consistent with the rest of the code. This overrides the guide.

## Working in this repo

- Run everything through uv: `uv run pytest`, `uv run python create_anonymization_plan.py --config ...`. A warning about an active conda `VIRTUAL_ENV` being ignored is harmless.
- Use the scratchpad for synthetic end-to-end runs (internal tooling only, see above). `tests/synthetic.py: build_tree(root)` writes a standard fake dataset. It includes multi-region clips, overlapping InstanceNumbers, a missing InstanceNumber, non-US, no-pixel and missing-key files, a corrupt file, and fake "PHI" strings (`synthetic.FAKE_PHI`) for leak tests.
- Script 2 can be exercised on CPU by calling `pending_records` → `shard_files` → `run_rank(rank, files, cfg, models)` with a stub `Models(echofine, echoprime, views, views)`. Raise a `BaseException` subclass to simulate a crash, since `Exception` is caught per record.
- This development node has **no GPU driver**. Real model loading and inference have never been run.
- `.gitignore` ignores `tests/`, `.claude/` and `.vscode/`. Those directories won't be committed.

## Current state and known issues (as of 2026-10-08)

- Script 1, Script 2, `merge`, `run_echo_preprocessing_and_anonymization.py`, the README and the run scripts are implemented.
- `run_echo_preprocessing_and_anonymization.py` was verified end-to-end on synthetic data (CPU): its rows are `preprocessed`; a rerun says "Nothing to process"; a later stub-model `run_rank` turns them into `success`. It has no test in `tests/` yet.
- **Column naming convention (user decision, 2026-10-08):** every column in every table (crosswalk, manifest, status, results, final outputs) is DICOM-style CamelCase, renamed at the source rather than at merge. Acronyms stay upper-case (`FPS`, `SOPClass`, `RegionsJSON`, `BBox2DX0`); suffixes are joined (`StudyDateShifted`, `PatientAgeCapped`). The view columns (`EchoFine_View=<view>`, `View_<Model>`, `ViewScore_<Model>`) keep their underscores by choice. The final outputs rename `AnonPatientID`/`AnonStudyID` → `PatientID`/`StudyID`, and `echo-record-list` renames `OutputPath` → `FilePath`. Only column names changed: Python dict keys inside `extract_header`, status/enum *values* (`excluded_sop_class`, `success`, …) and the keys inside `RegionsJSON` stay snake_case. Older names (snake_case `output_path`, `item_type`, `bbox_2d_x0`, `status`, `StudyDate_shifted`, …; before that `anon_patient`, `item_id`, …) are in outputs written before then. An old crosswalk fails the header check on load, and old manifest/status/result parts would raise a KeyError, so rerun both scripts.
- **Crosswalk rename (2026-10-08):** the crosswalk file is now `secure_dir/anonymization_crosswalk.csv` (was `crosswalk.csv`), and the module is `src/anon_crosswalk.py` (was `src/crosswalk.py`; class `Crosswalk` → `AnonCrosswalk`, `CROSSWALK_COLUMNS` → `ANON_CROSSWALK_COLUMNS`). Script 1 does not look for the old filename. A secure_dir from before the rename must have its `crosswalk.csv` renamed by hand, or Script 1 starts a new crosswalk at `P000000`/`S000000`.
- Internal `tests/test_index.py` covers Script 1: SOP filter, region records, collision naming, crosswalk restart and reuse, truncated-line recovery, crash/restart, and no PHI in the manifest. `tests/test_preprocess.py` and `tests/test_merge.py` cover Script 2 and `merge` with stub models; the new `echo-study-list.csv` has no test yet.
- `pyproject.toml` still has a `pytest` dev dependency group and `[tool.pytest.ini_options] testpaths = ["tests"]` for the internal tests.
- **Unfixed bug in `src/image_utils.py: crop_and_scale`:** when the aspect-ratio padding rounds to 0 (e.g. a 225×224 frame), `img[0:-0]` is empty and the function silently returns an all-black 224×224 image. The fix is to slice `img[p:img.shape[0]-p]` and do the same for width and zoom. It was reported to the user, who hasn't asked for it to be fixed yet.
- An image whose regions are all non-2D/1D (e.g. ECG only) gets null boxes and is saved fully black. That errs on the side of PHI safety, but the record is useless.
- VS Code may flag `tf_keras` as unresolved when it uses the conda interpreter; `.venv/bin/python` is the right one.
