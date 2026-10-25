from pathlib import Path

from speechopt.runtime import ActiveClock, lr_id


class FakeTime:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_idle_gap_between_sessions_is_not_counted(tmp_path: Path) -> None:
    clock_time = FakeTime(1000.0)
    path = tmp_path / "sessions.json"
    first = ActiveClock(path, now=clock_time)
    clock_time.t = 1100.0
    first.touch()  # ran for 100 s, then stopped
    clock_time.t = 1100.0 + 6 * 3600  # resumed six hours later
    second = ActiveClock(path, now=clock_time)
    clock_time.t += 50.0
    assert second.active_seconds() == 150.0


def test_crash_loses_only_time_since_last_touch(tmp_path: Path) -> None:
    clock_time = FakeTime(0.0)
    path = tmp_path / "sessions.json"
    c = ActiveClock(path, now=clock_time)
    clock_time.t = 60.0
    c.touch()
    clock_time.t = 90.0  # process dies here without touching
    clock_time.t = 500.0
    assert ActiveClock(path, now=clock_time).active_seconds() == 60.0


def test_sessions_persist_to_disk(tmp_path: Path) -> None:
    clock_time = FakeTime(10.0)
    path = tmp_path / "sessions.json"
    c = ActiveClock(path, now=clock_time)
    clock_time.t = 40.0
    c.touch()
    assert path.exists()
    import json

    assert json.loads(path.read_text()) == [[10.0, 40.0]]


def test_lr_folder_names() -> None:
    assert [lr_id(x) for x in (3e-3, 1e-2, 3e-2, 1e-4, 1.0, 0.1)] == [
        "3e-3",
        "1e-2",
        "3e-2",
        "1e-4",
        "1e00",
        "1e-1",
    ]
