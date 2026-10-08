# Echocardiography Examination Preprocessing & anonymization 

Two-stage pipeline that turns a large collection of echocardiogram DICOMs into:

1. a **study-level anonymization crosswalk** (`crosswalk.csv`, append-only),
2. **anonymized media**: `media/<anon_patient>/<anon_study>/<name>.<png|avi>`,
3. a **results table**: one row per media record with anonymized metadata and the outputs of two
   view classifiers (EchoFine, TF/Keras; EchoPrime, PyTorch).

| Script | Hardware | What it does |
|---|---|---|
| `index_and_anonymize.py` | CPU | Reads headers only, keeps ultrasound files, extends the crosswalk, finds ultrasound regions, writes the work manifest |
| `preprocess_and_classify.py run` | GPU | Decodes, crops/masks, runs both models, saves media, writes result parts |
| `preprocess_and_classify.py merge` | CPU | Combines result parts into the final anonymized table and prints QA counts |

Splitting the work this way lets you QA the anonymization before spending GPU time, and rerun Script 2 (new models or transforms) without touching the anonymization.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync                 # creates .venv with Python 3.11 and all dependencies (incl. CUDA wheels)
```

Put the model weights where `models:` in the config points (default `./weights/`).

## Configuration

Everything is set in **`configs/config.yaml`**. Usually you only edit `paths:`.

| Key | Meaning |
|---|---|
| `paths.dicom_root` | Raw DICOM root (searched recursively for `*.<dicom_extension>`) |
| `paths.secure_dir` | **Contains PHI**: `crosswalk.csv`, `manifest/`, `status/`, Script 1 logs. Restrict access. |
| `paths.output_root` | Anonymized outputs: `media/`, `results/`, `results_table.parquet`, Script 2 logs |
| `index.num_workers`, `index.chunk_size` | Header-reading pool size and files per chunk |
| `index.sop_classes` | SOP classes to keep (default: the two ultrasound storage classes) |
| `index.max_decoded_gb` | Files whose decoded RGB size exceeds this are flagged `oversize` and skipped by Script 2 |
| `index.date_offset_days` | Inclusive range for the per-patient random date shift (default +99 to +100 years) |
| `index.age_cap_years` | Ages at or above this become `"90+"` |
| `index.date_fields` | Dates copied after shifting, as `<Keyword>_shifted` |
| `index.metadata_allowlist` | **The only header fields copied.** Never add free text, names, institutions or private tags. |
| `preprocess.num_gpus` | One process per GPU. Uses the first N GPUs visible to the job (respects `CUDA_VISIBLE_DEVICES`). |
| `preprocess.num_workers`, `prefetch_factor` | DataLoader settings per GPU process |
| `preprocess.frame_chunk` | Frames per model call, so long clips can't exhaust VRAM |
| `preprocess.flush_every` | Result rows per part file |
| `preprocess.video_codec` | AVI FourCC (default `FFV1`, lossless) |
| `preprocess.pipeline_version` | Stored in every result row; bump it when models or preprocessing change |
| `models.*` | Weight paths, plus `echoprime_compile` (`torch.compile`) |

## Running

```bash
# 1. CPU node
scripts/run_anonymization_metadata.sh                 # = uv run python index_and_anonymize.py --config configs/config.yaml
#    -> QA secure_dir/crosswalk.csv, manifest/ and the run summary before going on

# 2. GPU node
scripts/run_anonymized_echos.sh                       # run, then merge
scripts/run_anonymized_echos.sh configs/config.yaml --retry-errors
# or the individual steps:
uv run python preprocess_and_classify.py run   --config configs/config.yaml [--retry-errors]
uv run python preprocess_and_classify.py merge --config configs/config.yaml
```

`run` exits with code 1 if any GPU process failed. Rerun it to resume.

## Outputs

### Secure area (`paths.secure_dir`), contains PHI

- **`crosswalk.csv`**: one row per study. The columns are `PatientID, PatientName, PatientSex, PatientAge, PatientBirthDate, AccessionNumber, StudyID, StudyInstanceUID, StudyDate, StudyTime, anon_patient, anon_study, date_offset_days`.
  - Keys: the patient key is `PatientID` and the study key is `StudyInstanceUID`.
  - IDs are sequential counters (`P000000`, `S000000`), with no hashing and no secret key.
  - A returning patient keeps their `anon_patient` and `date_offset_days`, so the intervals between their studies are preserved.
- **`manifest/chunk-NNNNNN.parquet`**: one row per media record. It includes the source DICOM path, so it must stay in the secure area. Each row has:
  - `item_id`, `output_path`, `anon_patient` and `anon_study`
  - `item_type` (image/video), `num_frames` and `fps`
  - region columns: `region_index`, the `bbox_*` / `bbox_2d_*` / `bbox_1d_*` boxes, `spatial_format`, `data_type`, `region_source` and `regions_json`
  - `decoded_bytes` and `oversize`
  - the allowlisted metadata, the `*_shifted` dates and `PatientAge_capped`
- **`status/chunk-NNNNNN.parquet`**: one row per discovered file, recording its `status`, `detail` and `n_records`. The possible statuses are:
  - `included`
  - `excluded_sop_class`, where `detail` holds the SOP class name; the run summary counts these by class
  - `excluded_no_pixel_data`
  - `excluded_missing_patient_id`
  - `excluded_missing_study_uid`
  - `excluded_no_video_region`: a video with no 2D tissue/color region
  - `error`, where `detail` holds the exception type only
- **`logs/`**: counts only. Logs never include tag values, file paths or exception messages.

### Anonymized area (`paths.output_root`)

- **`media/<anon_patient>/<anon_study>/<name>.png|.avi`**: preprocessed, PHI-masked media. Only the masked version is ever written.
- **`results/rank=<r>/part-NNNNNN.parquet`**: raw result rows, append-only. Each row has `item_id`, `output_path`, `status`, `error_type`, the `echofine_<view>` / `_std` / `echofine_view` columns, the `echoprime_<view>` / `_std` / `echoprime_view` columns, `pipeline_version` and `timestamp`.
  - `status` is one of `success`, `skipped_oversize`, `skipped_undecodable` or `error`.
- **`results_table.parquet`**: the final table, written by `merge`. It keeps the latest row per `item_id`, joined with the manifest through an explicit keep-list. It contains no source paths and no real identifiers.

## Output naming

Every `item_id` is `<anon_patient>/<anon_study>/<name>`. Script 1 assigns these names, in sorted file-path order. The first file to claim a name keeps it, and a name never changes once assigned.

| Case | Name |
|---|---|
| Normal | `<InstanceNumber>`, e.g. `12` |
| Video with several regions | every region gets `_r<region_index>`: `12_r0`, `12_r1` |
| Same InstanceNumber already used in the study (e.g. another series) | `_s<SeriesNumber>`: `12_s3`; if that is also taken, `12_s3-2`, `12_s3-3`, … |
| Missing InstanceNumber | `na0`, `na1`, … (next free in the study) |

## Region logic

Script 1:
- **Videos:** one record per region with `RegionSpatialFormat` = 2D and `RegionDataType` = tissue, color flow or unknown. A region whose box lies fully inside another kept region is dropped (e.g. a color box inside the sector).
  - No `SequenceOfUltrasoundRegions` → one full-frame record.
  - No qualifying region → `excluded_no_video_region`.
- **Images:** one record per file. `bbox_2d_*` is the largest 2D region and `bbox_1d_*` the largest M-mode or spectral region. Either may be empty. All regions are kept in `regions_json`.

Script 2:
- **EchoFine** gets the **full, unmasked frames**, resized to 224×224. It runs once per file, and all region records of that file share the result.
- **EchoPrime** gets the record's preprocessed media, resized to 224×224:
  - videos are cropped to the region box, then stationary pixels are masked;
  - images keep the full frame, with everything outside the 2D and 1D boxes zeroed.

## Restart behaviour

Both scripts can be killed at any point and rerun with the same command.

**Script 1**
- Each chunk is committed in this order:
  1. crosswalk rows are appended (then flushed and fsynced);
  2. the manifest part is written atomically;
  3. the status part is written atomically. This is the commit marker.
- Files already listed in a status part are skipped, including excluded and errored files, so they aren't rescanned.
- A manifest part without its status part is deleted and its chunk is redone.
- On restart, the chunk finds its studies already in the crosswalk, so no ID is ever assigned twice.
- A truncated last crosswalk line (crash mid-append) is dropped on startup. Counters resume from the highest existing IDs.

**Script 2**
- An `item_id` with any row in `results/` counts as done.
- A row is written only after both models have run and the media file has been saved (temp file + rename).
- A crash loses at most the unflushed rows. Those items are redone, and their media files are overwritten atomically.
- `--retry-errors` re-queues items whose latest status is `error`. `merge` keeps the latest row per `item_id`.

## Placeholders to fill in

| Where | What |
|---|---|
| `src/dicom_utils.py: find_us_regions(ds) -> list[Region]` | Region detection. The default reads `SequenceOfUltrasoundRegions`. Keep the signature and the half-open `(x0, y0, x1, y1)` box convention. |
| `preprocess_and_classify.py: AnonymizedEchoDataset.preprocess_echo_video` | Burned-in PHI removal for a video region. The default crops to the box and masks stationary pixels (`mask_outside_ultrasound_video`). |
| `preprocess_and_classify.py: AnonymizedEchoDataset.preprocess_echo_image` | Burned-in PHI removal for an image. The default zeroes everything outside the 2D and 1D boxes. |
| `configs/config.yaml: models` | Weight paths for EchoFine (`.h5`) and EchoPrime (`.pt`) |

## Code layout

```
index_and_anonymize.py      Script 1 (incl. output-naming rules)
preprocess_and_classify.py  Script 2: Dataset, GPU processes, `run` and `merge`
src/crosswalk.py            append-only crosswalk with crash recovery
src/dicom_utils.py          header reading, allowlist/date/age handling, regions, frame decoding
src/image_utils.py          masking, resizing, AVI writing
src/io_utils.py             config, atomic parquet writes, logging
src/view_classifier/        EchoFine / EchoPrime loaders and inference
```
