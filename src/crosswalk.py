"""Study-level anonymization crosswalk table 

One row per study. IDs are sequential counters (P000000, S000000) -- no secret key, no hashing.
Only the main process of Script 1 ever writes this file (single writer).

Crash safety: rows are appended one line each (embedded newlines are replaced by spaces before
writing), then flushed and fsynced. A crash mid-append can only leave a final line without a
trailing newline; on load that partial line is truncated away and the counters are restored from
the maximum existing IDs.
"""
import csv
import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

PATIENT_FIELDS = ["PatientID", "PatientName", "PatientSex", "PatientAge", "PatientBirthDate"]
STUDY_FIELDS = ["AccessionNumber", "StudyID", "StudyInstanceUID", "StudyDate", "StudyTime"]
CROSSWALK_COLUMNS = PATIENT_FIELDS + STUDY_FIELDS + ["anon_patient", "anon_study", "date_offset_days"]


def _format_id(prefix: str, n: int) -> str:
    return f"{prefix}{n:06d}"


def _parse_id(anon_id: str) -> int:
    return int(anon_id[1:])


def _clean(value) -> str:
    return "" if value is None else str(value).replace("\r", " ").replace("\n", " ")


class Crosswalk:
    def __init__(self, path, date_offset_range: tuple[int, int]):
        self.path = Path(path)
        self.date_offset_range = date_offset_range
        self.patients: dict[str, tuple[str, int]] = {}  # PatientID -> (anon_patient, date_offset_days)
        self.studies: dict[str, tuple[str, str, int]] = {}  # StudyInstanceUID -> (anon_patient, anon_study, offset)
        self.next_patient = 0
        self.next_study = 0
        self.key_conflicts = 0  # studies seen again under a different PatientID
        self._rng = secrets.SystemRandom()
        self._load()

    def _load(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._append_lines([CROSSWALK_COLUMNS])
            return

        # Drop a truncated final line left by a crash mid-append.
        data = self.path.read_bytes()
        if not data.endswith(b"\n"):
            keep = data.rfind(b"\n") + 1
            logger.warning("Crosswalk: dropping truncated final line (%d bytes)", len(data) - keep)
            with open(self.path, "r+b") as f:
                f.truncate(keep)
                f.flush()
                os.fsync(f.fileno())
            if keep == 0:  # even the header was partial
                self._append_lines([CROSSWALK_COLUMNS])
                return

        with open(self.path, newline="") as f:
            rows = list(csv.reader(f))
        if rows[0] != CROSSWALK_COLUMNS:
            raise ValueError("Crosswalk header does not match the expected columns")
        for line_no, row in enumerate(rows[1:], start=2):
            if len(row) != len(CROSSWALK_COLUMNS):
                # A complete (newline-terminated) line with the wrong field count is not a crash artifact.
                raise ValueError(f"Crosswalk line {line_no} has {len(row)} fields, expected {len(CROSSWALK_COLUMNS)}")
            rec = dict(zip(CROSSWALK_COLUMNS, row))
            offset = int(rec["date_offset_days"])
            self.patients.setdefault(rec["PatientID"], (rec["anon_patient"], offset))
            self.studies[rec["StudyInstanceUID"]] = (rec["anon_patient"], rec["anon_study"], offset)
            self.next_patient = max(self.next_patient, _parse_id(rec["anon_patient"]) + 1)
            self.next_study = max(self.next_study, _parse_id(rec["anon_study"]) + 1)
        logger.info("Crosswalk loaded: %d patients, %d studies", len(self.patients), len(self.studies))

    def _append_lines(self, rows: list[list[str]]) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())

    def assign(self, study_records: list[dict]) -> int:
        """Assign IDs to unseen patients/studies and durably append their rows.

        `study_records` are dicts holding the crosswalk source fields, in file order. For a study
        appearing several times, the first occurrence supplies the patient/study fields. New IDs are
        handed out in sorted (PatientID, StudyInstanceUID) order so the result is deterministic.
        Returns the number of new rows appended.
        """
        first_seen: dict[str, dict] = {}
        for rec in study_records:
            uid = rec["StudyInstanceUID"]
            if uid in self.studies:
                if self.patients.get(rec["PatientID"], (None,))[0] != self.studies[uid][0]:
                    self.key_conflicts += 1
                continue
            if uid in first_seen:
                if first_seen[uid]["PatientID"] != rec["PatientID"]:
                    self.key_conflicts += 1
                continue
            first_seen[uid] = rec

        new_rows = []
        for rec in sorted(first_seen.values(), key=lambda r: (r["PatientID"], r["StudyInstanceUID"])):
            pid = rec["PatientID"]
            if pid not in self.patients:
                offset = self._rng.randint(*self.date_offset_range)
                self.patients[pid] = (_format_id("P", self.next_patient), offset)
                self.next_patient += 1
            anon_patient, offset = self.patients[pid]
            anon_study = _format_id("S", self.next_study)
            self.next_study += 1
            self.studies[rec["StudyInstanceUID"]] = (anon_patient, anon_study, offset)
            new_rows.append([_clean(rec.get(c)) for c in PATIENT_FIELDS + STUDY_FIELDS]
                            + [anon_patient, anon_study, str(offset)])
        if new_rows:
            self._append_lines(new_rows)
        return len(new_rows)

    def lookup(self, study_uid: str) -> tuple[str, str, int]:
        """(anon_patient, anon_study, date_offset_days) for an already-assigned study."""
        return self.studies[study_uid]
