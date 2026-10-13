"""Persistent JSON response cache for Wikidata requests."""

import json
import sqlite3
from pathlib import Path
from typing import Any


class WikidataCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT)"
        )

    def get(self, key: str) -> Any:
        row = self._db.execute(
            "SELECT body FROM responses WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def store(self, key: str, body: Any) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?)",
                (key, json.dumps(body)),
            )

    def close(self) -> None:
        self._db.close()
