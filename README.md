# Echocardiography preprocessing & anonymization 

## Introduction

Two-stage pipeline that turns a large collection of echocardiogram DICOMs into:

1. a **study-level anonymization crosswalk** (`anonymization_crosswalk.csv`, append-only),
2. **anonymized media**: `files/<AnonPatientID>/<AnonStudyID>/<name>.<png|avi>`,
3. a **record list** (`echo-record-list.parquet`): one row per media record with anonymized metadata and the outputs of two
   view classifiers,
4. a **study list** (`echo-study-list.csv`): one row per study.

| Script | Hardware | What it does |
|---|---|---|
| `create_anonymization_plan.py` (Script 1) | CPU | Reads headers only, keeps ultrasound files, extends the crosswalk, finds ultrasound regions, writes the work manifest |
| `run_echo_preprocessing_and_anonymization_with_classifier.py run` (Script 2) | GPU | Decodes, crops/masks, runs both models, saves media, writes result parts |
| `run_echo_preprocessing_and_anonymization.py` | CPU | Same as `run`, but without the view classifiers: decodes, crops/masks, saves media, writes result parts. For when no GPU is available. |
| `run_echo_preprocessing_and_anonymization_with_classifier.py merge` | CPU | Combines result parts into the final anonymized table and prints QA counts |

Splitting the work this way lets you QA the anonymization before spending GPU time, and rerun Script 2 (new models or transforms) without touching the anonymization: bump `preprocess.pipeline_version`
and every record is processed again.

### Metadata anonymization

Script 1 (`create_anonymization_plan.py`) reads only the DICOM headers and keeps files whose SOP class is ultrasound (`index.sop_classes`). Everything that
can identify a patient stays in the secure area (`paths.secure_dir`); only de-identified data reaches `paths.output_root`.

- **Identifiers.** Each patient (`PatientID`) and study (`StudyInstanceUID`) gets a sequential ID (`P000000`, `S000000`), with
  no hashing and no secret key. The mapping is stored in the append-only crosswalk, so a returning patient keeps their
  `AnonPatientID`. Files missing either key are excluded, never grouped with other files.
- **Dates.** Each patient gets one random offset of +99 to +100 years (`index.date_offset_days`), reused for all of their
  studies, so the intervals between studies are preserved. The dates in `index.date_fields` are copied only after shifting,
  as `<Keyword>Shifted`.
- **Age.** Ages at or above `index.age_cap_years` (default 90) become `"<age_cap_years>+"`, e.g. `"90+"`
  (`PatientAgeCapped`).
- **Other fields.** Only the header fields in `index.metadata_allowlist` are copied. Names, free text, institutions and
  private tags are never copied.
- **Final tables.** `merge` builds the record and study lists through an explicit keep-list, so source paths and crosswalk
  fields never reach them.
- **Logs** record counts and exception types only, never tag values or source DICOM paths. Paths under
  `paths.output_root` (e.g. where the final tables were written) can appear. pydicom's own log messages and warnings
  are silenced in every process that reads DICOMs. An unhandled crash still prints a Python traceback to stderr (and
  so to the Slurm job output).

### Video and image anonymization

Burned-in PHI (names, dates, IDs and other text overlaid on the frames) is removed from the pixels. Only the masked
version of each file is ever written to disk.

**Regions (Script 1).** The regions come from the DICOM `SequenceOfUltrasoundRegions`:

- **Videos:** one record per region with `RegionSpatialFormat` = 2D and `RegionDataType` = tissue, color flow or unknown. A
  region whose box lies fully inside another kept region is dropped (e.g. a color box inside the sector).
  - No `SequenceOfUltrasoundRegions` → one full-frame record.
  - No qualifying region → excluded (`excluded_no_video_region`).
- **Images:** one record per file, holding at most two boxes. `BBox2D*` is the largest 2D region with
  `RegionDataType` = tissue, color flow or unknown (the same rule as videos), and `BBox1D*` the largest M-mode or spectral
  region (ties go to the lower region index). Either may be empty. If an image has several 2D
  or several 1D regions, only the largest of each kind is kept for masking; the others (e.g. the second panel of a
  side-by-side layout) are masked out. All regions are still listed in `RegionsJSON`.
  - No `SequenceOfUltrasoundRegions` → excluded (`excluded_no_image_region`), since nothing would say what to keep.
  - Neither box (e.g. only ECG or graphics regions) → excluded (`excluded_no_image_region`).

**Masking (Script 2).** Masked pixels are set to 0 (black).

- **Videos** (one output clip per region record):
  1. Crop the clip to the region box. Everything outside the box is dropped.
  2. Build one mask for the whole clip from pixel motion: a pixel is kept if it is non-black on average and varies over
     time. The variation map is eroded first, so small changes such as the ECG trace or updating heart-rate text don't
     count as motion. If several separate areas remain, the largest is taken as the ultrasound sector and filled out to
     its convex hull.
  3. Zero every pixel outside the sector in every frame. This removes static overlays (names, dates, labels, the ECG
     trace).
  4. Crop the clip to the sector's bounding box.

  The motion mask is adapted from
  [EchoPrime](https://github.com/echonet/EchoPrime/blob/main/video_utils.py) (`mask_outside_ultrasound_video`).
- **Images** have no motion to use, so the boxes alone define what is kept. The output keeps the full frame size, and
  every pixel outside `BBox2D ∪ BBox1D` is zeroed. Pixels inside the boxes are kept as they are, so any text drawn
  inside a box is not removed.

<!-- TODO: replace with an example of a masked video (e.g. a GIF of original vs. anonymized). -->
![Video masking example: original (left) and anonymized (right)](docs/figures/video_masking_example.gif)

*Figure 1. Video masking example. (placeholder)*

<!-- TODO: replace with an example of a masked image (e.g. an M-mode or Doppler still). -->
![Image masking example: original (left) and anonymized (right)](docs/figures/image_masking_example.png)

*Figure 2. Image masking example. (placeholder)*

### View classification

Script 2 labels each record with its echocardiographic view using two classifiers:

| Model | Framework | Views | Input | Paper |
|---|---|---|---|---|
| EchoFine | TF/Keras | 47 | The **full, unmasked frames**, center-cropped to a square and resized to 224×224. It runs once per file, and all region records of that file share the result. | [EchoFine paper](TODO-link-to-paper) |
| EchoPrime | PyTorch | 11 | The record's masked media, center-cropped to a square and resized to 224×224 | [EchoPrime paper](TODO-link-to-paper) |

Each model's per-frame probabilities are reduced to one score per view, `score = mean / (1 + std)` over frames, so views
that are confident and consistent across the clip score highest. The top-scoring view is stored as the record's view (see
the result columns under [Outputs](#outputs)). `run_echo_preprocessing_and_anonymization.py` skips this step.

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
| `paths.dicom_root` | Raw DICOM root (searched recursively) |
| `paths.dicom_extension` | Extension of the files to pick up (default `dcm`, case-insensitive) |
| `paths.secure_dir` | **Contains PHI**: `anonymization_crosswalk.csv`, `manifest/`, `status/`, Script 1 logs. Restrict access. |
| `paths.output_root` | Anonymized outputs: `files/`, `results/`, `echo-record-list.parquet`, `echo-study-list.csv`, Script 2 logs |
| `index.num_workers`, `index.chunk_size` | Header-reading pool size and files per chunk |
| `index.sop_classes` | SOP classes to keep (default: the two ultrasound storage classes) |
| `index.max_num_frames` | Files with more frames than this are excluded (`excluded_too_many_frames`); `null` = no limit |
| `index.max_decoded_gb` | Files whose decoded RGB size exceeds this are flagged `Oversize` and skipped by Script 2 (`skipped_oversize`) |
| `index.date_offset_days` | Inclusive range for the per-patient random date shift (default +99 to +100 years) |
| `index.age_cap_years` | Ages at or above this become `"<age_cap_years>+"` |
| `index.default_fps` | Frame rate used when a video has no `CineRate`, `FrameTime` or `FrameTimeVector` |
| `index.date_fields` | Dates copied after shifting, as `<Keyword>Shifted`. Keep `StudyDate` in it: the study list's `StudyDateTime` is built from `StudyDateShifted` and is empty without it. |
| `index.metadata_allowlist` | **The only header fields copied.** Never add free text, names, institutions or private tags. |
| `preprocess.num_gpus` | One process per GPU. Uses the first N GPUs visible to the job (respects `CUDA_VISIBLE_DEVICES`). |
| `preprocess.num_cpu_procs` | Number of processes for `run_echo_preprocessing_and_anonymization.py` (each also starts `num_workers` DataLoader workers) |
| `preprocess.num_workers`, `prefetch_factor` | DataLoader settings per GPU (or CPU) process |
| `preprocess.frame_chunk` | Frames per model call, so long clips can't exhaust VRAM |
| `preprocess.flush_every` | Result rows per part file |
| `preprocess.video_codec` | AVI FourCC (default `FFV1`, lossless) |
| `preprocess.pipeline_version` | Stored in every result row. Bump it when models or preprocessing change: records whose latest result has a different version are processed again. |
| `models.*` | Weight paths, plus `echoprime_compile` (`torch.compile`) |

## Running

```bash
# 1. CPU node
scripts/run_anonymization_metadata.sh                 # = uv run python create_anonymization_plan.py --config configs/config.yaml
#    -> QA secure_dir/anonymization_crosswalk.csv, manifest/ and the run summary before going on

# 2. GPU node
scripts/run_anonymized_echos.sh                       # run, then merge
scripts/run_anonymized_echos.sh configs/config.yaml --retry-errors
# or the individual steps:
uv run python run_echo_preprocessing_and_anonymization_with_classifier.py run   --config configs/config.yaml [--retry-errors]
uv run python run_echo_preprocessing_and_anonymization_with_classifier.py merge --config configs/config.yaml
```

`run` exits with code 1 if any GPU process failed. Rerun it to resume.

### Without a GPU (preprocessing only)

When no GPU is available, you can skip the view classifiers and only preprocess and save the media:

```bash
scripts/run_preprocess_only.sh                        # preprocess only, then merge
scripts/run_preprocess_only.sh configs/config.yaml --retry-errors
# or the individual steps:
uv run python run_echo_preprocessing_and_anonymization.py --config configs/config.yaml [--retry-errors]
uv run python run_echo_preprocessing_and_anonymization_with_classifier.py merge --config configs/config.yaml
```

- It runs on CPU with `preprocess.num_cpu_procs` processes and never loads the models.
- It writes the same media and result parts as `run`, but the rows have status `preprocessed` and no view columns.
- A later `run_echo_preprocessing_and_anonymization_with_classifier.py run` on a GPU node treats `preprocessed` records as pending. It redoes them with classification and overwrites the same media files.
- Don't run `run_echo_preprocessing_and_anonymization.py` and `run_echo_preprocessing_and_anonymization_with_classifier.py run` at the same time. They write to the same `results/rank=<r>/` directories.
- `run_echo_preprocessing_and_anonymization.py` exits with code 1 if any process failed. Rerun it to resume.

### On a Slurm cluster

`slurm/` has example batch scripts that wrap the same commands. Submit them from the repo root:

```bash
slurm/submit_pipeline.sh                              # Script 1, then Script 2 (GPU) once Script 1 succeeds
# or one stage at a time:
sbatch slurm/01_create_anonymization_plan.sbatch [configs/config.yaml]
sbatch slurm/02_run_with_classifier.sbatch       [configs/config.yaml] [--retry-errors]   # GPU run, then merge
sbatch slurm/02_run_preprocess_only.sbatch       [configs/config.yaml] [--retry-errors]   # CPU only, then merge
```

- When you call `sbatch` directly, run `mkdir -p slurm/logs` first (it's gitignored, and Slurm can't write the job output without it). `submit_pipeline.sh` does this for you.
- Before you submit, set the partition and account in the `#SBATCH` lines, and add a `module load` if `uv` isn't on the compute nodes' `PATH`.
- The scripts can't read the config, so keep their resource requests in step with it by hand:
  - `--gres=gpu:N` must be at least `preprocess.num_gpus`.
  - `--cpus-per-task` should cover `index.num_workers` for Script 1, and `num_gpus` (or `num_cpu_procs`) × (`num_workers` + 1) for Script 2.
- Job output goes to `slurm/logs/<job-name>_<job-id>.out`. Like the pipeline's own logs, it holds only counts and exception types.
- Every stage resumes where it stopped, so if a job hits its time limit, resubmit it.

## Outputs

### Secure area (`paths.secure_dir`), contains PHI

- **`anonymization_crosswalk.csv`**: one row per study. The columns are `PatientID, PatientName, PatientSex, PatientAge, PatientBirthDate, AccessionNumber, StudyID, StudyInstanceUID, StudyDate, StudyTime, AnonPatientID, AnonStudyID, DateOffsetDays`.
- **`manifest/chunk-NNNNNN.parquet`**: one row per media record. It includes the source DICOM path, so it must stay in the secure area. Each row has:
  - `RecordID`, `OutputPath`, `SourcePath`, `AnonPatientID` and `AnonStudyID`
  - `ItemType` (image/video), `SOPClass`, `NumFrames` and `FPS`
  - region columns: `RegionIndex`, the `BBox*` / `BBox2D*` / `BBox1D*` boxes (`X0`, `Y0`, `X1`, `Y1`), `SpatialFormat`, `DataType`, `RegionSource` and `RegionsJSON`
  - `DecodedBytes` and `Oversize`
  - the allowlisted metadata, the `*Shifted` dates and `PatientAgeCapped`
- **`status/chunk-NNNNNN.parquet`**: one row per discovered file, recording its `SourcePath`, `Status`, `Detail` and `NumRecords`. The possible statuses are:
  - `included`
  - `excluded_sop_class`, where `Detail` holds the SOP class name; the run summary counts these by class
  - `excluded_no_pixel_data`
  - `excluded_missing_patient_id`
  - `excluded_missing_study_uid`
  - `excluded_too_many_frames`: more frames than `index.max_num_frames`
  - `excluded_no_video_region`: a video with no 2D tissue/color region
  - `excluded_no_image_region`: an image with no region sequence, or with no 2D tissue/color region and no M-mode/spectral
    region
  - `error`, where `Detail` holds the exception type only
- **`logs/`**: counts and exception types only.

### Anonymized area (`paths.output_root`)

- **`files/<AnonPatientID>/<AnonStudyID>/<name>.png|.avi`**: preprocessed, PHI-masked media.
- **`logs/`**: Script 2 and `merge` logs (counts and exception types only).
- **`results/rank=<r>/part-NNNNNN.parquet`**: raw result rows, append-only. Each row has `RecordID`, `OutputPath`, `Status`, `ErrorType`, the view-classification columns (see below), `PipelineVersion` and `Timestamp`.
  - `Status` is one of `success`, `preprocessed` (media saved by `run_echo_preprocessing_and_anonymization.py`, not classified), `skipped_oversize`, `skipped_undecodable` or `error`.
  - View classification (on `success` rows only, see [View classification](#view-classification)):
    - `EchoFine_View=<view>` / `EchoPrime_View=<view>`: the score for each view.
    - `View_EchoFine` / `View_EchoPrime`: the top-scoring view.
    - `ViewScore_EchoFine` / `ViewScore_EchoPrime`: that view's score.
- **`echo-record-list.parquet`**: the final table, written by `merge`. It keeps the latest row per `RecordID`, joined with the manifest through an explicit keep-list. Rows of every status are kept (check `Status`; only `success` rows have view columns), and manifest records with no result yet are not in it. `AnonPatientID` / `AnonStudyID` are renamed to `PatientID` / `StudyID` here, since the final outputs hold no real IDs, and `OutputPath` is renamed to `FilePath` (relative to `files/`).
- **`echo-study-list.csv`**: written by `merge`, one row per study in the record list, with columns `PatientID` (the anonymized `AnonPatientID`), `StudyID` (the anonymized `AnonStudyID`) and `StudyDateTime` (the shifted StudyDate plus the StudyTime; date only if StudyTime is missing).

## Output naming

Every `RecordID` is `<AnonPatientID>/<AnonStudyID>/<name>`. Script 1 assigns these names, in sorted file-path order. The first file to claim a name keeps it, and a name never changes once assigned.

| Case | Name |
|---|---|
| Normal | `<InstanceNumber>`, e.g. `12` |
| Video with several regions | every region gets `_r<region_index>`: `12_r0`, `12_r1` |
| Same InstanceNumber already used in the study (e.g. another series) | `_s<SeriesNumber>`: `12_s3`; if that is also taken, `12_s3-2`, `12_s3-3`, … |
| Missing InstanceNumber | `na0`, `na1`, … (next free in the study) |

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
- A `RecordID` whose latest row in `results/` has the current `pipeline_version` counts as done, whatever its status. The exception is `preprocessed`: it counts as done for `run_echo_preprocessing_and_anonymization.py` but as pending for `run`. A row from another `pipeline_version` counts as pending.
- A row is written only after both models have run and the media file has been saved (temp file + rename).
- A crash loses at most the unflushed rows. Those items are redone, and their media files are overwritten atomically.
- `--retry-errors` re-queues items whose latest status is `error`. `merge` keeps the latest row per `RecordID`.
- `merge` deletes `results/` once the merge is complete, meaning every manifest record has a result from the current `pipeline_version`, none is `error`, and no media file is missing. Otherwise it keeps `results/` so that `run` can resume. After `results/` is deleted, a later `run` treats every record as pending again.

## Tests

```bash
uv run pytest
```

The tests use only fake DICOMs generated by `tests/synthetic.py` and stub view classifiers, so they need no real data,
model weights or GPU.

## Placeholders to fill in

| Where | What |
|---|---|
| `configs/config.yaml: models` | Weight paths for EchoFine (`.h5`) and EchoPrime (`.pt`) |

## Code layout

```
create_anonymization_plan.py                                 Script 1 (incl. output-naming rules)
run_echo_preprocessing_and_anonymization_with_classifier.py  Script 2: Dataset, GPU processes, `run` and `merge`
run_echo_preprocessing_and_anonymization.py                  Script 2 without the view classifiers (CPU processes)
src/anon_crosswalk.py                                        append-only crosswalk with crash recovery
src/dicom_utils.py                                           header reading, allowlist/date/age handling, regions, frame decoding
src/image_utils.py                                           masking, resizing, AVI writing
src/io_utils.py                                              config, atomic parquet writes, logging
src/view_classifier/                                         EchoFine / EchoPrime loaders and inference
scripts/                                                     shell wrappers for each stage
slurm/                                                       example sbatch scripts for each stage, plus submit_pipeline.sh
tests/                                                       pytest suite on synthetic DICOMs (no real data, no GPU)
```
