"""Append-only CSV journals with bounded recovery and explicit compaction."""

from __future__ import annotations

import json
import os
import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from planning.common.io import write_csv_atomic


JOURNAL_CONTRACT_ID = "csv_journal_v1"


class CsvJournalError(RuntimeError):
    """Raised when a journal cannot be recovered without guessing."""


class CsvJournal:
    """Persist rows append-only and compact them to CSV at an explicit boundary.

    The journal has one schema record followed by one JSON row record per append.
    Each append flushes Python's file buffer but never calls ``fsync``.  Durable
    synchronization is reserved for ``flush(fsync=True)`` and ``commit``.
    """

    def __init__(
        self,
        path: Path,
        *,
        fieldnames: Optional[Sequence[str]] = None,
        key_field: Optional[str] = None,
        retain_rows: bool = True,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = tuple(str(value) for value in fieldnames or ())
        self._key_field = str(key_field) if key_field else ""
        self._retain_rows = bool(retain_rows)
        self._rows: List[Dict] = []
        self._keys: Dict[str, Dict] = {}
        self._row_count = 0
        self._last_key = None
        self._last_row = None
        self._terminal_valid_bytes = None
        self._handle = None
        self._closed = False
        if self._retain_rows:
            self._load()
        else:
            self._load_streaming()
        self._handle = self.path.open("a", encoding="utf-8")

    @property
    def fieldnames(self) -> tuple:
        return self._fieldnames

    @property
    def key_field(self) -> str:
        return self._key_field

    @property
    def committed_rows(self) -> List[Dict]:
        """Return a snapshot of rows accepted by the journal."""

        if self._retain_rows:
            return [dict(row) for row in self._rows]
        return [dict(row) for row in self.iter_rows()]

    @property
    def row_count(self) -> int:
        return int(self._row_count)

    @property
    def last_row(self) -> Optional[Dict]:
        return None if self._last_row is None else dict(self._last_row)

    def _read_records(self):
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        payload = self.path.read_bytes()
        raw_lines = payload.splitlines(keepends=True)
        self._terminal_valid_bytes = len(payload)
        records = []
        for index, raw_line in enumerate(raw_lines):
            if not raw_line.strip():
                continue
            if not raw_line.endswith((b"\n", b"\r")) and index == len(raw_lines) - 1:
                # A process may have been interrupted between write and newline.
                self._terminal_valid_bytes = len(payload) - len(raw_line)
                break
            try:
                records.append(json.loads(raw_line.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CsvJournalError(
                    "invalid non-terminal journal record at {}:{}".format(
                        self.path, index + 1
                    )
                ) from error
        return records

    def _load(self) -> None:
        records = self._read_records()
        if (
            self._terminal_valid_bytes is not None
            and self._terminal_valid_bytes < self.path.stat().st_size
        ):
            with self.path.open("r+b") as handle:
                handle.truncate(self._terminal_valid_bytes)
        if not records:
            return
        header = records[0]
        if header.get("_journal") != JOURNAL_CONTRACT_ID:
            raise CsvJournalError("journal contract mismatch: {}".format(self.path))
        stored_fields = tuple(str(value) for value in header.get("fieldnames", ()))
        stored_key = str(header.get("key_field", ""))
        if self._fieldnames and self._fieldnames != stored_fields:
            raise CsvJournalError(
                "journal field contract mismatch: expected={} actual={}".format(
                    self._fieldnames, stored_fields
                )
            )
        if self._key_field and self._key_field != stored_key:
            raise CsvJournalError(
                "journal key contract mismatch: expected={} actual={}".format(
                    self._key_field, stored_key
                )
            )
        self._fieldnames = stored_fields
        self._key_field = stored_key
        for index, record in enumerate(records[1:], start=2):
            if not isinstance(record, dict) or not isinstance(record.get("_row"), dict):
                raise CsvJournalError(
                    "invalid row record at {}:{}".format(self.path, index)
                )
            self._accept_row(record["_row"], persist=False)

    def _load_streaming(self) -> None:
        """Recover a streaming journal without reading its full byte payload."""

        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        valid_bytes = 0
        file_size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            header_line = handle.readline()
            if not header_line or not header_line.endswith((b"\n", b"\r")):
                raise CsvJournalError("journal header is truncated: {}".format(self.path))
            try:
                header = json.loads(header_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CsvJournalError("invalid journal header: {}".format(self.path)) from error
            if header.get("_journal") != JOURNAL_CONTRACT_ID:
                raise CsvJournalError("journal contract mismatch: {}".format(self.path))
            stored_fields = tuple(str(value) for value in header.get("fieldnames", ()))
            stored_key = str(header.get("key_field", ""))
            if self._fieldnames and self._fieldnames != stored_fields:
                raise CsvJournalError(
                    "journal field contract mismatch: expected={} actual={}".format(
                        self._fieldnames, stored_fields
                    )
                )
            if self._key_field and self._key_field != stored_key:
                raise CsvJournalError(
                    "journal key contract mismatch: expected={} actual={}".format(
                        self._key_field, stored_key
                    )
                )
            self._fieldnames = stored_fields
            self._key_field = stored_key
            valid_bytes = len(header_line)
            for line_number, raw_line in enumerate(handle, start=2):
                if not raw_line.strip():
                    valid_bytes += len(raw_line)
                    continue
                if not raw_line.endswith((b"\n", b"\r")):
                    break
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise CsvJournalError(
                        "invalid non-terminal journal record at {}:{}".format(
                            self.path, line_number
                        )
                    ) from error
                if not isinstance(record, dict) or not isinstance(record.get("_row"), dict):
                    raise CsvJournalError(
                        "invalid row record at {}:{}".format(self.path, line_number)
                    )
                self._accept_row(record["_row"], persist=False)
                valid_bytes += len(raw_line)
        if valid_bytes < file_size:
            with self.path.open("r+b") as handle:
                handle.truncate(valid_bytes)

    def _ensure_open(self) -> None:
        if self._closed:
            raise CsvJournalError("journal is closed: {}".format(self.path))

    def _ensure_schema(self, row: Dict) -> None:
        if not self._fieldnames:
            self._fieldnames = tuple(str(key) for key in row.keys())
        if not self._key_field:
            self._key_field = "episode_id" if "episode_id" in row else ""
        if self._key_field and self._key_field not in self._fieldnames:
            raise CsvJournalError(
                "journal key field missing from schema: {}".format(self._key_field)
            )
        missing = [field for field in self._fieldnames if field not in row]
        if missing:
            raise CsvJournalError(
                "journal row missing fields {} at {}".format(missing, self.path)
            )

    def _accept_row(self, row: Dict, *, persist: bool) -> bool:
        if not isinstance(row, dict):
            raise CsvJournalError("journal row must be a mapping")
        row = {str(key): value for key, value in row.items()}
        self._ensure_schema(row)
        normalized = {field: row.get(field, "") for field in self._fieldnames}
        key = None
        if self._key_field:
            key = str(normalized[self._key_field])
            if not key:
                raise CsvJournalError("journal row has empty key {}".format(self._key_field))
            if self._retain_rows:
                previous = self._keys.get(key)
                if previous is not None:
                    if previous != normalized:
                        raise CsvJournalError(
                            "conflicting duplicate journal key {}={!r}".format(
                                self._key_field, key
                            )
                        )
                    return False
                self._keys[key] = dict(normalized)
        if self._retain_rows:
            self._rows.append(dict(normalized))
        else:
            if self._key_field:
                if self._last_key == key and self._last_row == normalized:
                    return False
                self._last_key = key
            self._last_row = dict(normalized)
        self._row_count += 1
        if persist:
            self._write_header_if_needed()
            self._handle.write(
                json.dumps(
                    {"_row": normalized},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            self._handle.flush()
        return True

    def _write_header_if_needed(self) -> None:
        if self.path.stat().st_size != 0:
            return
        self._handle.write(
            json.dumps(
                {
                    "_journal": JOURNAL_CONTRACT_ID,
                    "fieldnames": list(self._fieldnames),
                    "key_field": self._key_field,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    def append(self, row: Dict) -> bool:
        """Append a row; identical duplicate keys are idempotent."""

        self._ensure_open()
        return self._accept_row(row, persist=True)

    def import_rows(self, rows: Iterable[Dict]) -> int:
        """Seed a new journal from a legacy CSV checkpoint exactly once."""

        imported = 0
        for row in rows:
            imported += int(self.append(dict(row)))
        return imported

    def flush(self, *, fsync: bool = False) -> None:
        self._ensure_open()
        self._handle.flush()
        if fsync:
            os.fsync(self._handle.fileno())

    def recover(self) -> List[Dict]:
        """Flush and return the recovered, de-duplicated row sequence."""

        self.flush()
        fieldnames = self._fieldnames
        key_field = self._key_field
        self._rows = []
        self._keys = {}
        self._row_count = 0
        self._last_key = None
        self._last_row = None
        self._fieldnames = fieldnames
        self._key_field = key_field
        if self._retain_rows:
            self._load()
        else:
            self._load_streaming()
        return self.committed_rows

    def compact(self, output_path: Path) -> int:
        """Write one atomic CSV snapshot without closing the journal."""

        self.flush()
        if not self._retain_rows:
            target = Path(output_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(self._fieldnames), extrasaction="ignore"
                )
                writer.writeheader()
                count = 0
                for row in self.iter_rows():
                    writer.writerow(row)
                    count += 1
            os.replace(str(temporary), str(target))
            return count
        write_csv_atomic(
            Path(output_path),
            self._rows,
            fieldnames=list(self._fieldnames),
        )
        return len(self._rows)

    def iter_rows(self) -> Iterator[Dict]:
        """Stream committed row records without retaining the full history."""

        self.flush()
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CsvJournalError(
                        "invalid journal record at {}:{}".format(self.path, line_number)
                    ) from error
                if record.get("_journal") == JOURNAL_CONTRACT_ID:
                    continue
                row = record.get("_row")
                if not isinstance(row, dict):
                    raise CsvJournalError(
                        "invalid row record at {}:{}".format(self.path, line_number)
                    )
                yield {str(key): row.get(key, "") for key in self._fieldnames}

    def commit(self, output_path: Path) -> int:
        """Synchronize the journal and publish one final CSV snapshot."""

        self.flush(fsync=True)
        return self.compact(output_path)

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._handle.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
