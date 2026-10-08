# AGENTS.md — repository context for coding agents

Context for AI agents working in this repo. Behavioural rules for agents are in `.claude/CLAUDE.md`. User-facing docs (config keys, run commands, output layout) are in `README.md`; this file doesn't repeat them.

## What this repo is

A two-stage pipeline that de-identifies a large (>1M file) echocardiogram DICOM dataset and classifies each record's view.

1. **`index_and_anonymize.py`** (Script 1, CPU) reads headers only, without decoding pixels.
   - Keeps only the SOP classes `Ultrasound Image Storage` and `Ultrasound Multi-frame Image Storage`.
   - Extends an append-only study-level **crosswalk** CSV.
   - Finds ultrasound regions.
   - Writes an item-level **work manifest** (Parquet parts) plus a per-file **status log**.
2. **`preprocess_and_classify.py`** (Script 2, GPU) has two subcommands:
   - `run` decodes each pending manifest record, crops and masks it, runs **EchoFine** (TF/Keras, 47 views) and **EchoPrime** (PyTorch, 11 views), saves PNG/AVI media under `output_root/file/`, and appends result parts. It spawns one process per GPU.
   - `merge` builds the final anonymized `echo-record-list.parquet` and `echo-study-list.csv` (`subject_id`, `study_id`, `study_datetime`). For the study list it reads only the `anon_study` and `StudyTime` columns of the crosswalk; `study_datetime` = shifted StudyDate + unshifted StudyTime.

All settings live in `configs/config.yaml`, which both scripts share. The user is meant to edit only this file.

## Hard rules

- **Never open, print or log real DICOM headers or pixel data on this server.** Develop and test only against `tests/synthetic.py` fixtures or pydicom test data. Do not run either script against real paths.
- **Logs and status rows record the exception *type* only**, never messages, tag values or file paths. pydicom's logger and warnings are silenced via `quiet_pydicom()` for the same reason.
- **The crosswalk is append-only.** Existing IDs must never change, and counters resume from the highest existing ID.
  - IDs are sequential, with no hashing and no secret key: `P000000` per `PatientID`, `S000000` per `StudyInstanceUID`.
  - Only Script 1's main process writes the crosswalk. Workers just return the keys they read.
- **Metadata leaves the secure area only through allowlists:**
  - `index.metadata_allowlist` in the config;
  - `MANIFEST_KEEP` / `RESULT_KEEP` in `preprocess_and_classify.py`, which `merge` uses as a keep-list rather than a drop-list.
  - `source_path` and the crosswalk fields must never reach `output_root`.
- **The manifest and status parts contain source paths, so they are PHI** and live in `paths.secure_dir`. Anything under `paths.output_root` must be anonymized.

## Layout

```
index_and_anonymize.py      Script 1. Also holds the output-naming functions (assign_stem, item_id_for, stem_of).
                            find_files_by_extension is the user's helper; keep it.
preprocess_and_classify.py  Script 2: AnonymizedEchoDataset, Models/load_models, run_rank, ResultWriter,
                            pending_records, shard_files, run, merge, cli
src/crosswalk.py            Crosswalk class: load + truncated-last-line recovery, assign(), append+fsync
src/dicom_utils.py          read_header (read_partial, stop at pixel data; reports has_pixels), sop_class_name,
                            to_primitive, compute_fps, shift_date, cap_age, extract_header (worker entry),
                            Region + find_us_regions + select_video_regions/select_image_regions,
                            decode_frames (frame-level iter_pixels -> (N,H,W,3) uint8 RGB)
src/image_utils.py          (pre-existing, user's) save_video_cv2, mask_outside_ultrasound_video, crop_and_scale
src/io_utils.py             load_config, atomic_write_parquet (tmp + fsync + os.replace), read_parquet_dir, setup_logging
src/view_classifier/        (pre-existing, user's) echofine.py (imports TF at module level), echoprime.py
configs/config.yaml         the only config
scripts/*.sh                uv run wrappers for the two stages (gitignored)
slurm/                      empty, gitignored
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
  - the naming registry = the stems of all manifest `item_id`s.
- Part numbers form one global sequence across runs. Don't put timestamps in part names: a same-second restart once overwrote a committed chunk.
- Status values: `included`, `excluded_sop_class` (detail = SOP name), `excluded_no_pixel_data`, `excluded_missing_patient_id`, `excluded_missing_study_uid`, `excluded_no_video_region`, `error` (detail = exception type).

**Script 2**
- Done = every `item_id` in `output_root/results/rank=*/part-*.parquet`, whatever its status. With `--retry-errors`, items whose latest row is `error` are re-queued.
- Pending records are grouped **by source file** and sorted by `decoded_bytes` (largest first), then dealt round-robin to the ranks.
- The parent sets `CUDA_VISIBLE_DEVICES` before each `spawn` and maps rank r to the r-th GPU already visible to the job. TF is imported only inside `load_models` and enables memory growth.
- DataLoader: `batch_size=None`, identity collate, `multiprocessing_context="spawn"`, `in_order=False`, `persistent_workers=True`.
- A result row is added only after both models have run **and** the media file has been saved atomically. The temp name keeps the extension (`x.tmp.avi`) because cv2/PIL choose the format from it.
- Result statuses: `success`, `skipped_oversize`, `skipped_undecodable`, `error`.
- `__main__` re-imports the module as `preprocess_and_classify`, so the Dataset and process targets pickle under the real module name for spawned children. Keep that.
- `run` returns False and the CLI exits 1 if any rank process fails.

## Decisions confirmed by the user (don't change without asking)

- **Output naming** (`<anon_patient>/<anon_study>/<name>`, first claimer wins, a name never changes):
  - The base is the InstanceNumber.
  - Every record of a multi-region video gets `_r<region_index>`.
  - A later clash in the same study gets `_s<SeriesNumber>`, or `_sna` if SeriesNumber is missing; a further clash adds `-2`, `-3`, ….
  - A missing InstanceNumber gives `na<k>` (next free k).
  - The registry stores stems, i.e. the item_id without `_r<i>`.
- **Video regions:** keep regions with RegionSpatialFormat = 2D and RegionDataType ∈ {unknown/0, Tissue, Color Flow}.
  - Drop a region whose box lies fully inside another kept region.
  - No region sequence → one full-frame record (`region_source="full_frame"`).
  - No qualifying region → `excluded_no_video_region`.
- **Images:** one record each.
  - `bbox_2d` = the largest 2D region; `bbox_1d` = the largest M-mode/Spectral region. Either may be null.
  - All regions go into `regions_json`.
  - No region sequence → `bbox_2d` is the full frame.
- **Boxes are half-open** `(x0, y0, x1, y1)`, ready for numpy slicing. DICOM's MaxX1/MaxY1 are inclusive, so 1 is added.
- **Missing PatientID or StudyInstanceUID** → the file is excluded with a status. It is never grouped with other files.
- **Date shift:** one random integer offset per patient, in [36159, 36525] days (+99–100 y). It is drawn via `secrets.SystemRandom` and reused for all of that patient's studies. Ages of 90 or more become `"90+"`.
- **Model inputs:**
  - EchoFine gets the **full, unmasked frames** (`crop_and_scale(..., zoom=0)` to 224). It runs once per source file, and all region records of that file share its output.
  - EchoPrime gets the record's masked media, resized to 224.
- **Only the masked media is written to disk**, at native (cropped) resolution:
  - Videos are cropped to the region box, then `mask_outside_ultrasound_video(single_mask=True, crop_to_mask=True)`.
  - Images keep the full frame, with everything outside `bbox_2d ∪ bbox_1d` zeroed.
- **Environment:** uv with `python-preference = "only-managed"`, Python 3.11, `.venv`. torch/torchvision come from the cu128 index; the dependencies include `tensorflow[and-cuda]` and `tf_keras`.

## Placeholders the user will replace

Keep these signatures stable:
- `src/dicom_utils.py: find_us_regions(ds) -> list[Region]`
- `AnonymizedEchoDataset.preprocess_echo_video(video, bbox)`
- `AnonymizedEchoDataset.preprocess_echo_image(image, bbox_2d, bbox_1d)`

## Working in this repo

- Run everything through uv: `uv run pytest`, `uv run python index_and_anonymize.py --config ...`. A warning about an active conda `VIRTUAL_ENV` being ignored is harmless.
- Use the scratchpad for synthetic end-to-end runs (internal tooling only, see above). `tests/synthetic.py: build_tree(root)` writes a standard fake dataset. It includes multi-region clips, overlapping InstanceNumbers, a missing InstanceNumber, non-US, no-pixel and missing-key files, a corrupt file, and fake "PHI" strings (`synthetic.FAKE_PHI`) for leak tests.
- Script 2 can be exercised on CPU by calling `pending_records` → `shard_files` → `run_rank(rank, files, cfg, models)` with a stub `Models(echofine, echoprime, views, views)`. Raise a `BaseException` subclass to simulate a crash, since `Exception` is caught per record.
- This development node has **no GPU driver**. Real model loading and inference have never been run.
- `.gitignore` ignores `tests/`, `scripts/`, `slurm/`, `.claude/` and `.vscode/`. Those directories won't be committed.

## Current state and known issues (as of 2026-10-08)

- Script 1, Script 2, `merge`, the README and the run scripts are implemented.
- Internal `tests/test_index.py` covers Script 1: SOP filter, region records, collision naming, crosswalk restart and reuse, truncated-line recovery, crash/restart, and no PHI in the manifest. `tests/test_preprocess.py` and `tests/test_merge.py` cover Script 2 and `merge` with stub models; the new `echo-study-list.csv` has no test yet.
- `pyproject.toml` still has a `pytest` dev dependency group and `[tool.pytest.ini_options] testpaths = ["tests"]` for the internal tests.
- **Unfixed bug in `src/image_utils.py: crop_and_scale`:** when the aspect-ratio padding rounds to 0 (e.g. a 225×224 frame), `img[0:-0]` is empty and the function silently returns an all-black 224×224 image. The fix is to slice `img[p:img.shape[0]-p]` and do the same for width and zoom. It was reported to the user, who hasn't asked for it to be fixed yet.
- An image whose regions are all non-2D/1D (e.g. ECG only) gets null boxes and is saved fully black. That errs on the side of PHI safety, but the record is useless.
- VS Code may flag `tf_keras` as unresolved when it uses the conda interpreter; `.venv/bin/python` is the right one.
