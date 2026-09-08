from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from planning.data.csv_journal import CsvJournal, CsvJournalError


FIELDS = ("episode_id", "status")


def test_csv_journal_recovers_and_compacts_without_per_row_fsync(tmp_path, monkeypatch):
    journal_path = tmp_path / "rows.journal.jsonl"
    output_path = tmp_path / "rows.csv"
    fsync_calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_calls.append(fd))

    journal = CsvJournal(journal_path, fieldnames=FIELDS, key_field="episode_id")
    assert journal.append({"episode_id": "0", "status": "ok"}) is True
    assert journal.append({"episode_id": "0", "status": "ok"}) is False
    assert fsync_calls == []
    assert journal.committed_rows == [{"episode_id": "0", "status": "ok"}]
    assert journal.commit(output_path) == 1
    assert fsync_calls
    journal.close()

    recovered = CsvJournal(journal_path, fieldnames=FIELDS, key_field="episode_id")
    assert recovered.recover() == [{"episode_id": "0", "status": "ok"}]
    assert output_path.read_text(encoding="utf-8").splitlines()[0] == "episode_id,status"
    recovered.close()


def test_csv_journal_ignores_truncated_terminal_record(tmp_path):
    path = tmp_path / "rows.journal.jsonl"
    journal = CsvJournal(path, fieldnames=FIELDS, key_field="episode_id")
    journal.append({"episode_id": "1", "status": "ok"})
    journal.close()
    with path.open("ab") as handle:
        handle.write(b'{"_row":{"episode_id":"2"')
    recovered = CsvJournal(path, fieldnames=FIELDS, key_field="episode_id")
    assert recovered.committed_rows == [{"episode_id": "1", "status": "ok"}]
    recovered.close()


def test_csv_journal_rejects_conflicting_duplicate(tmp_path):
    journal = CsvJournal(
        tmp_path / "rows.journal.jsonl", fieldnames=FIELDS, key_field="episode_id"
    )
    journal.append({"episode_id": "1", "status": "ok"})
    with pytest.raises(CsvJournalError, match="conflicting duplicate"):
        journal.append({"episode_id": "1", "status": "failed"})
    journal.close()


def test_csv_journal_imports_legacy_rows_once(tmp_path):
    path = tmp_path / "rows.journal.jsonl"
    journal = CsvJournal(path, fieldnames=FIELDS, key_field="episode_id")
    assert journal.import_rows(
        [{"episode_id": "1", "status": "ok"}, {"episode_id": "2", "status": "ok"}]
    ) == 2
    assert journal.import_rows([{"episode_id": "1", "status": "ok"}]) == 0
    journal.close()


def test_csv_journal_streaming_mode_keeps_only_bounded_recovery_state(tmp_path):
    path = tmp_path / "streaming.journal.jsonl"
    output = tmp_path / "streaming.csv"
    journal = CsvJournal(
        path,
        fieldnames=FIELDS,
        key_field="episode_id",
        retain_rows=False,
    )
    for index in range(10000):
        assert journal.append({"episode_id": str(index), "status": "ok"}) is True

    assert journal.row_count == 10000
    assert journal._rows == []
    assert journal._keys == {}
    assert journal.last_row == {"episode_id": "9999", "status": "ok"}
    assert journal.compact(output) == 10000
    journal.close()

    recovered = CsvJournal(
        path,
        fieldnames=FIELDS,
        key_field="episode_id",
        retain_rows=False,
    )
    assert recovered.row_count == 10000
    assert recovered._rows == []
    assert recovered._keys == {}
    assert sum(1 for _ in recovered.iter_rows()) == 10000
    recovered.close()
