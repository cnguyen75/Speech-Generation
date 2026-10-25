"""Bookkeeping for the resumable study that must not depend on models or data."""

import json
from pathlib import Path
import time

from .common import write_json


class ActiveClock:
    """Counts only the time the job was actually running, across stops and restarts.

    Each process start appends a [start, last_seen] session to a JSON file; `touch()` moves
    last_seen forward (the study calls it on every log line, roughly once a minute). Idle gaps
    between sessions are never counted. After a crash, only the time since the last touch is lost.
    """

    def __init__(self, path, now=time.time):
        self.path, self.now = Path(path), now
        self.sessions = json.loads(self.path.read_text()) if self.path.exists() else []
        t = self.now()
        self.sessions.append([t, t])
        self._save()

    def _save(self):
        write_json(self.path, self.sessions)

    def touch(self):
        self.sessions[-1][1] = self.now()
        self._save()

    def active_seconds(self):
        self.touch()
        return sum(end - start for start, end in self.sessions)


def lr_id(lr):
    """Folder-name form of a learning rate: 0.003 -> '3e-3', 1.0 -> '1e00'."""
    return f"{lr:.0e}".replace("+", "").replace("-0", "-")
