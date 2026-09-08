from __future__ import annotations

from collections import namedtuple

from planning.teacher import collection_disk
from planning.teacher.collection_disk import DiskSpaceWatch


def test_disk_capacity_gate_accounts_for_target_and_margin(tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        collection_disk.shutil,
        "disk_usage",
        lambda path: usage(10 * collection_disk.GIB, 7 * collection_disk.GIB, 3 * collection_disk.GIB),
    )
    result = collection_disk.disk_capacity_gate(
        tmp_path,
        target_accepted=100,
        minimum_free_gb=1.0,
        safety_margin_gb=0.1,
        estimated_episode_bytes=1024 * 1024,
    )
    assert result.passed is True
    assert result.estimate.target_accepted == 100
    assert result.estimate.estimated_episode_bytes == 1024 * 1024
    assert result.estimate.sampled_episode_count == 0
    assert result.required_free_bytes < result.snapshot.free_bytes


def test_disk_capacity_gate_fails_before_runtime_when_space_is_insufficient(
    tmp_path, monkeypatch
):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        collection_disk.shutil,
        "disk_usage",
        lambda path: usage(10 * collection_disk.GIB, 9 * collection_disk.GIB, 1 * collection_disk.GIB),
    )
    result = collection_disk.disk_capacity_gate(
        tmp_path,
        target_accepted=2000,
        minimum_free_gb=0.5,
        safety_margin_gb=0.5,
        estimated_episode_bytes=1024 * 1024,
    )
    assert result.passed is False
    assert result.reason == "INSUFFICIENT_FREE_SPACE"


def test_disk_space_watch_emits_threshold_transitions(tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    free_values = iter((4.0, 2.0, 0.5, 0.5))

    def fake_disk_usage(path):
        free_gb = next(free_values)
        return usage(
            10 * collection_disk.GIB,
            int((10.0 - free_gb) * collection_disk.GIB),
            int(free_gb * collection_disk.GIB),
        )

    monkeypatch.setattr(
        collection_disk.shutil,
        "disk_usage",
        fake_disk_usage,
    )
    watch = DiskSpaceWatch(
        tmp_path,
        warning_free_gb=3.0,
        stop_free_gb=1.0,
        interval_s=1.0,
    )
    assert watch.check(force=True).level == "OK"
    assert watch.check(force=True).level == "WARNING"
    assert watch.check(force=True).level == "STOP"
    assert watch.check(force=True) is None
