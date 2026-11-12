"""Persist local chat conversations and messages in SQLite."""

import json
import sqlite3
from pathlib import Path
from typing import Literal, TypedDict


DB_PATH = Path(__file__).with_name("chat_history.sqlite3")


class Conversation(TypedDict):
    id: int
    title: str
    updated_at: str


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    sources: list[dict[str, str]]


def _connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open a database connection and ensure its parent directory exists."""
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database_path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_chat_history(db_path: str | Path = DB_PATH) -> None:
    """Create the conversation and message tables when they do not exist."""
    connection = _connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%d %H:%M:%f', 'now')
                ),
                updated_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%d %H:%M:%f', 'now')
                )
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%d %H:%M:%f', 'now')
                )
            );
            CREATE INDEX IF NOT EXISTS messages_conversation_id_idx
                ON messages(conversation_id, id);
            """
        )
        connection.commit()
    finally:
        connection.close()


def create_conversation(db_path: str | Path = DB_PATH) -> int:
    """Create a new empty conversation and return its database ID."""
    initialize_chat_history(db_path)
    connection = _connect(db_path)
    try:
        cursor = connection.execute("INSERT INTO conversations DEFAULT VALUES")
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def list_conversations(db_path: str | Path = DB_PATH) -> list[Conversation]:
    """Return saved conversations with the most recently active first."""
    initialize_chat_history(db_path)
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id, title, updated_at
            FROM conversations
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "id": int(row["id"]),
            "title": str(row["title"]),
            "updated_at": str(row["updated_at"]),
        }
        for row in rows
    ]


def load_messages(
    conversation_id: int,
    db_path: str | Path = DB_PATH,
) -> list[ChatMessage]:
    """Load a conversation's messages in their original order."""
    initialize_chat_history(db_path)
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT role, content, sources_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()
    finally:
        connection.close()
    messages: list[ChatMessage] = []
    for row in rows:
        sources = json.loads(row["sources_json"])
        if not isinstance(sources, list):
            raise ValueError("Saved chat sources must be a list.")
        messages.append(
            {
                "role": row["role"],
                "content": str(row["content"]),
                "sources": sources,
            }
        )
    return messages


def save_message(
    conversation_id: int,
    role: Literal["user", "assistant"],
    content: str,
    sources: list[dict[str, str]] | None = None,
    db_path: str | Path = DB_PATH,
) -> None:
    """Save a message and title a conversation from its first user question."""
    if not content.strip():
        raise ValueError("Chat messages cannot be empty.")
    initialize_chat_history(db_path)
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    connection = _connect(db_path)
    try:
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, sources_json)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, sources_json),
        )
        title = " ".join(content.split())
        if role == "user":
            title = title[:77].rstrip()
            if len(" ".join(content.split())) > 77:
                title += "..."
            connection.execute(
                """
                UPDATE conversations
                SET title = CASE WHEN title = 'New chat' THEN ? ELSE title END,
                    updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
                WHERE id = ?
                """,
                (title, conversation_id),
            )
        else:
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
                WHERE id = ?
                """,
                (conversation_id,),
            )
        connection.commit()
        if cursor.rowcount != 1:
            raise sqlite3.IntegrityError("The chat message could not be saved.")
    finally:
        connection.close()


def clear_conversation(conversation_id: int, db_path: str | Path = DB_PATH) -> None:
    """Remove a conversation's messages while keeping it available as a new chat."""
    connection = _connect(db_path)
    try:
        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = 'New chat',
                updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
            WHERE id = ?
            """,
            (conversation_id,),
        )
        connection.commit()
        if cursor.rowcount != 1:
            raise sqlite3.IntegrityError("The chat could not be cleared.")
    finally:
        connection.close()


def rename_conversation(
    conversation_id: int,
    title: str,
    db_path: str | Path = DB_PATH,
) -> None:
    """Rename one conversation without changing its messages or identifier."""
    normalized_title = " ".join(title.split())
    if not normalized_title:
        raise ValueError("Conversation titles cannot be empty.")

    initialize_chat_history(db_path)
    connection = _connect(db_path)
    try:
        cursor = connection.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (normalized_title, conversation_id),
        )
        connection.commit()
        if cursor.rowcount != 1:
            raise sqlite3.IntegrityError("The conversation could not be renamed.")
    finally:
        connection.close()


def delete_conversation(conversation_id: int, db_path: str | Path = DB_PATH) -> None:
    """Permanently delete a conversation and its messages."""
    initialize_chat_history(db_path)
    connection = _connect(db_path)
    try:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        connection.commit()
        if cursor.rowcount != 1:
            raise sqlite3.IntegrityError("The chat could not be deleted.")
    finally:
        connection.close()
