"""Verify that the Streamlit home page renders without loading Torch."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppStartupTests(unittest.TestCase):
    def test_home_page_does_not_load_embedding_model(self) -> None:
        with (
            patch(
                "chat_history.list_conversations",
                return_value=[
                    {"id": 1, "title": "Test chat", "updated_at": "now"}
                ],
            ),
            patch("chat_history.load_messages", return_value=[]),
            patch(
                "vector_store.load_vector_store",
                return_value=(faiss.IndexFlatL2(384), []),
            ),
            patch("embeddings._load_sentence_transformer") as load_model,
        ):
            app = AppTest.from_file(str(APP_PATH)).run()

        self.assertFalse(app.exception, app.exception)
        self.assertTrue(
            any("Web RAG Assistant" in element.value for element in app.markdown)
        )
        load_model.assert_not_called()
        self.assertNotIn("sentence_transformers", sys.modules)
        self.assertNotIn("torch", sys.modules)
