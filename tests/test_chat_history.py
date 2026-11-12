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
    rename_conversation,
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

    def test_rename_updates_only_the_selected_conversation_and_preserves_messages(self) -> None:
        first_id = create_conversation(self.database_path)
        second_id = create_conversation(self.database_path)
        save_message(first_id, "user", "First chat", db_path=self.database_path)
        save_message(
            first_id,
            "assistant",
            "An answer with a source.",
            [{"title": "Example", "url": "https://example.com"}],
            db_path=self.database_path,
        )
        save_message(second_id, "user", "Second chat", db_path=self.database_path)
        original_messages = load_messages(first_id, self.database_path)

        rename_conversation(first_id, "  Research   notes  ", self.database_path)

        conversations = {
            conversation["id"]: conversation["title"]
            for conversation in list_conversations(self.database_path)
        }
        self.assertEqual(conversations[first_id], "Research notes")
        self.assertEqual(conversations[second_id], "Second chat")
        self.assertEqual(load_messages(first_id, self.database_path), original_messages)
        self.assertEqual(load_messages(second_id, self.database_path)[0]["content"], "Second chat")

    def test_rename_allows_duplicate_titles_without_changing_the_other_chat(self) -> None:
        first_id = create_conversation(self.database_path)
        second_id = create_conversation(self.database_path)

        rename_conversation(first_id, "Shared title", self.database_path)
        rename_conversation(second_id, "Shared title", self.database_path)

        self.assertEqual(
            {chat["id"] for chat in list_conversations(self.database_path)},
            {first_id, second_id},
        )
        self.assertEqual(
            [chat["title"] for chat in list_conversations(self.database_path)],
            ["Shared title", "Shared title"],
        )

    def test_rename_rejects_empty_titles_without_modifying_conversation(self) -> None:
        conversation_id = create_conversation(self.database_path)
        save_message(conversation_id, "user", "Existing title", db_path=self.database_path)

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            rename_conversation(conversation_id, " \n\t ", self.database_path)

        self.assertEqual(list_conversations(self.database_path)[0]["title"], "Existing title")

    def test_rename_missing_conversation_reports_error(self) -> None:
        with self.assertRaisesRegex(
            sqlite3.IntegrityError,
            "could not be renamed",
        ):
            rename_conversation(999, "Missing chat", self.database_path)

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
