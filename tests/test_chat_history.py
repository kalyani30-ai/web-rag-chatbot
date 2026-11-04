"""Tests for locally persisted SQLite chat conversations."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from chat_history import (
    clear_conversation,
    create_conversation,
    delete_conversation,
    list_conversations,
    load_messages,
    save_message,
)


class ChatHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "history.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_saves_and_reopens_messages_with_sources(self) -> None:
        conversation_id = create_conversation(self.database_path)
        sources = [{"title": "Example", "url": "https://example.com"}]
        save_message(conversation_id, "user", "What is this?", db_path=self.database_path)
        save_message(
            conversation_id,
            "assistant",
            "A grounded answer.",
            sources,
            db_path=self.database_path,
        )

        self.assertEqual(
            load_messages(conversation_id, self.database_path),
            [
                {"role": "user", "content": "What is this?", "sources": []},
                {
                    "role": "assistant",
                    "content": "A grounded answer.",
                    "sources": sources,
                },
            ],
        )
        self.assertEqual(
            list_conversations(self.database_path)[0]["title"],
            "What is this?",
        )

    def test_conversations_are_independent_and_listed_recently_updated(self) -> None:
        first_id = create_conversation(self.database_path)
        second_id = create_conversation(self.database_path)
        save_message(first_id, "user", "First chat", db_path=self.database_path)
        save_message(second_id, "user", "Second chat", db_path=self.database_path)

        conversations = list_conversations(self.database_path)
        self.assertEqual(
            {conversation["id"] for conversation in conversations},
            {first_id, second_id},
        )
        self.assertEqual(load_messages(first_id, self.database_path)[0]["content"], "First chat")
        self.assertEqual(load_messages(second_id, self.database_path)[0]["content"], "Second chat")

    def test_clear_removes_messages_and_resets_title(self) -> None:
        conversation_id = create_conversation(self.database_path)
        save_message(conversation_id, "user", "Old conversation", db_path=self.database_path)

        clear_conversation(conversation_id, self.database_path)

        self.assertEqual(load_messages(conversation_id, self.database_path), [])
        self.assertEqual(list_conversations(self.database_path)[0]["title"], "New chat")

    def test_delete_removes_conversation_and_its_messages(self) -> None:
        conversation_id = create_conversation(self.database_path)
        other_conversation_id = create_conversation(self.database_path)
        save_message(conversation_id, "user", "Remove this chat", db_path=self.database_path)
        save_message(
            conversation_id,
            "assistant",
            "This answer should be removed too.",
            db_path=self.database_path,
        )

        delete_conversation(conversation_id, self.database_path)

        self.assertEqual(
            [chat["id"] for chat in list_conversations(self.database_path)],
            [other_conversation_id],
        )
        self.assertEqual(load_messages(conversation_id, self.database_path), [])

    def test_delete_missing_conversation_reports_error(self) -> None:
        with self.assertRaisesRegex(
            sqlite3.IntegrityError,
            "The chat could not be deleted",
        ):
            delete_conversation(999, self.database_path)


if __name__ == "__main__":
    unittest.main()
