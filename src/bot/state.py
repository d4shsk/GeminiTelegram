import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import HISTORY_KEEP_MESSAGES, HISTORY_TOKEN_BUDGET, MAX_HISTORY, SUMMARY_CHAR_BUDGET


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _normalize_text(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


class _PersistentChatField:
    def __init__(self, state: "BotState", column: str) -> None:
        self.state = state
        self.column = column

    def get(self, chat_id: int, default: Any = None) -> Any:
        return self.state._get_setting(chat_id, self.column, default)

    def __setitem__(self, chat_id: int, value: str) -> None:
        self.state._set_setting(chat_id, self.column, value)

    def pop(self, chat_id: int, default: Any = None) -> Any:
        value = self.get(chat_id, default)
        self.state._clear_setting(chat_id, self.column)
        return value


@dataclass
class BotState:
    db_path: Path = field(default_factory=lambda: BotState._resolve_db_path())
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _connection: sqlite3.Connection | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.user_modes = _PersistentChatField(self, "user_mode")
        self.user_models = _PersistentChatField(self, "user_model")
        self.active_models = _PersistentChatField(self, "active_model")
        self.initialize()

    @staticmethod
    def _resolve_db_path() -> Path:
        env_path = os.environ.get("BOT_DB_PATH")
        if env_path:
            path = Path(env_path)
        else:
            mount_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
            if mount_path:
                path = Path(mount_path) / "bot_state.sqlite3"
            else:
                path = Path("data") / "bot_state.sqlite3"

        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def initialize(self) -> None:
        with self._lock:
            if self._connection is None:
                self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
                self._connection.row_factory = sqlite3.Row
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_settings (
                    chat_id INTEGER PRIMARY KEY,
                    user_mode TEXT,
                    user_model TEXT,
                    active_model TEXT
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id_id
                ON chat_messages(chat_id, id);

                CREATE TABLE IF NOT EXISTS chat_summaries (
                    chat_id INTEGER PRIMARY KEY,
                    summary TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_profiles (
                    chat_id INTEGER PRIMARY KEY,
                    profile_json TEXT NOT NULL
                );
                """
            )
            self._connection.commit()

    def _conn(self) -> sqlite3.Connection:
        self.initialize()
        assert self._connection is not None
        return self._connection

    def _get_setting(self, chat_id: int, column: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn().execute(
                f"SELECT {column} FROM chat_settings WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            if not row or row[column] is None:
                return default
            return row[column]

    def _set_setting(self, chat_id: int, column: str, value: str) -> None:
        with self._lock:
            self._conn().execute(
                f"""
                INSERT INTO chat_settings (chat_id, {column})
                VALUES (?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET {column} = excluded.{column}
                """,
                (chat_id, value),
            )
            self._conn().commit()

    def _clear_setting(self, chat_id: int, column: str) -> None:
        with self._lock:
            self._conn().execute(
                f"UPDATE chat_settings SET {column} = NULL WHERE chat_id = ?",
                (chat_id,),
            )
            self._conn().commit()

    def get_history(self, chat_id: int) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn().execute(
                "SELECT role, text FROM chat_messages WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [{"role": row["role"], "text": row["text"]} for row in rows]

    def get_summary(self, chat_id: int) -> str:
        with self._lock:
            row = self._conn().execute(
                "SELECT summary FROM chat_summaries WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        return row["summary"] if row else ""

    def get_user_profile(self, chat_id: int) -> dict[str, Any]:
        with self._lock:
            row = self._conn().execute(
                "SELECT profile_json FROM chat_profiles WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["profile_json"])
        except json.JSONDecodeError:
            return {}

    def update_user_profile(self, chat_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        if not updates:
            return self.get_user_profile(chat_id)

        with self._lock:
            profile = self.get_user_profile(chat_id)
            merged = {**profile, **updates}
            for key, value in list(merged.items()):
                if value is None or value == [] or value == "":
                    merged.pop(key, None)
            self._conn().execute(
                """
                INSERT INTO chat_profiles (chat_id, profile_json)
                VALUES (?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET profile_json = excluded.profile_json
                """,
                (chat_id, json.dumps(merged, ensure_ascii=False, sort_keys=True)),
            )
            self._conn().commit()
            return merged

    def update_history(self, chat_id: int, role: str, text: str) -> None:
        if not text:
            return

        with self._lock:
            self._conn().execute(
                "INSERT INTO chat_messages (chat_id, role, text) VALUES (?, ?, ?)",
                (chat_id, role, text),
            )
            self._compact_history(chat_id)
            self._conn().commit()

    def clear_history(self, chat_id: int) -> None:
        with self._lock:
            self._conn().execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
            self._conn().execute("DELETE FROM chat_summaries WHERE chat_id = ?", (chat_id,))
            self._conn().execute("UPDATE chat_settings SET active_model = NULL WHERE chat_id = ?", (chat_id,))
            self._conn().commit()

    def _fetch_messages(self, chat_id: int) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT id, role, text FROM chat_messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
        return [{"id": row["id"], "role": row["role"], "text": row["text"]} for row in rows]

    def _set_summary(self, chat_id: int, summary: str) -> None:
        if summary:
            self._conn().execute(
                """
                INSERT INTO chat_summaries (chat_id, summary)
                VALUES (?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET summary = excluded.summary
                """,
                (chat_id, summary),
            )
        else:
            self._conn().execute("DELETE FROM chat_summaries WHERE chat_id = ?", (chat_id,))

    def _delete_message_ids(self, message_ids: list[int]) -> None:
        if not message_ids:
            return
        placeholders = ",".join("?" for _ in message_ids)
        self._conn().execute(f"DELETE FROM chat_messages WHERE id IN ({placeholders})", message_ids)

    def _compact_history(self, chat_id: int) -> None:
        messages = self._fetch_messages(chat_id)
        if not messages:
            self._set_summary(chat_id, "")
            return

        if len(messages) > MAX_HISTORY:
            overflow = messages[:-MAX_HISTORY]
            self._delete_message_ids([message["id"] for message in overflow])
            messages = messages[-MAX_HISTORY:]

        summary = self.get_summary(chat_id)
        while self._context_tokens(summary, messages) > HISTORY_TOKEN_BUDGET and len(messages) > HISTORY_KEEP_MESSAGES:
            fold_count = max(1, len(messages) - HISTORY_KEEP_MESSAGES)
            folded = messages[:fold_count]
            summary = self._merge_summary(summary, folded)
            self._delete_message_ids([message["id"] for message in folded])
            messages = messages[fold_count:]

        while self._context_tokens(summary, messages) > HISTORY_TOKEN_BUDGET and len(messages) > 2:
            folded = messages[:1]
            summary = self._merge_summary(summary, folded)
            self._delete_message_ids([message["id"] for message in folded])
            messages = messages[1:]

        self._set_summary(chat_id, summary)

    def _context_tokens(self, summary: str, messages: list[dict[str, Any]]) -> int:
        total = _estimate_tokens(summary) if summary else 0
        for message in messages:
            total += _estimate_tokens(message["text"])
        return total

    def _merge_summary(self, existing_summary: str, messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        if existing_summary:
            lines.append(existing_summary)

        for message in messages:
            speaker = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {_normalize_text(message['text'])}")

        summary = "Summary of earlier conversation:\n" + "\n".join(f"- {line}" for line in lines)
        if len(summary) <= SUMMARY_CHAR_BUDGET:
            return summary

        trimmed_lines: list[str] = []
        current_size = len("Summary of earlier conversation:\n")
        for line in reversed(lines):
            entry = f"- {line}"
            projected = current_size + len(entry) + 1
            if projected > SUMMARY_CHAR_BUDGET:
                break
            trimmed_lines.append(entry)
            current_size = projected

        trimmed_lines.reverse()
        return "Summary of earlier conversation:\n" + "\n".join(trimmed_lines)


state = BotState()
