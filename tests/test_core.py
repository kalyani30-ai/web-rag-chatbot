"""Focused tests for retrieval, URL handling, indexing, and Groq response parsing."""

import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import faiss
import numpy as np
import requests

from groq_client import (
    GroqClientError,
    generate_answer,
    groq_api_key_configured,
)
from api_keys import get_api_key
from embeddings import EmbeddingModelError, get_embedding_model
from processor import split_into_chunks
from prompt_builder import build_rag_prompt
from rag_pipeline import retrieve_relevant_chunks, update_knowledge_base
from scraper import (
    WebsiteScrapeError,
    normalize_website_url,
    scrape_website_with_metadata,
    validate_website_url,
)
from search import build_search_query, search_web
from vector_store import (
    append_to_vector_store,
    create_vector_store,
    load_vector_store,
    search_vector_store,
)


class CountingEmbeddingModel:
    """A tiny predictable model used to verify one query encoding per search."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_texts: list[str] = []
        self.convert_to_numpy = False

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        self.calls += 1
        self.last_texts = texts
        self.convert_to_numpy = convert_to_numpy
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class CoreBehaviorTests(unittest.TestCase):
    def test_retrieval_encodes_once_and_preserves_sources(self) -> None:
        index = create_vector_store(
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )
        chunks = [
            {"text": "first source", "url": "https://one.example/a", "title": "One"},
            {"text": "second source", "url": "https://two.example/b", "title": "Two"},
        ]
        model = CountingEmbeddingModel()

        retrieved = retrieve_relevant_chunks(
            "question",
            index,
            chunks,
            top_k=8,
            embedding_model=model,
        )

        self.assertEqual(model.calls, 1)
        self.assertEqual(model.last_texts, ["question"])
        self.assertTrue(model.convert_to_numpy)
        self.assertEqual(len(retrieved), 2)
        self.assertEqual(retrieved[0]["url"], chunks[0]["url"])
        self.assertEqual(retrieved[0]["title"], "One")

    def test_retrieval_preserves_uploaded_file_source_name(self) -> None:
        index = create_vector_store(np.asarray([[1.0, 0.0]], dtype=np.float32))
        chunks = [
            {
                "text": "uploaded file content",
                "url": "uploaded://hash/report.pdf",
                "title": "report.pdf",
                "file_name": "report.pdf",
            }
        ]

        retrieved = retrieve_relevant_chunks(
            "question",
            index,
            chunks,
            embedding_model=CountingEmbeddingModel(),
        )

        self.assertEqual(retrieved[0]["file_name"], "report.pdf")

    def test_retrieval_reuses_provided_question_embedding(self) -> None:
        index = create_vector_store(np.asarray([[1.0, 0.0]], dtype=np.float32))
        model = CountingEmbeddingModel()

        retrieved = retrieve_relevant_chunks(
            "question",
            index,
            [{"text": "source", "url": "https://example.com"}],
            embedding_model=model,
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        )

        self.assertEqual(model.calls, 0)
        self.assertEqual(retrieved[0]["text"], "source")

    def test_search_caps_top_k_to_available_vectors(self) -> None:
        index = create_vector_store(np.asarray([[1.0, 0.0]], dtype=np.float32))
        distances, positions = search_vector_store(
            index,
            np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=4,
        )
        self.assertEqual(positions.shape, (1, 1))
        self.assertEqual(float(distances[0][0]), 0.0)

    def test_chunker_rejects_an_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            split_into_chunks("some text", chunk_size=4, overlap=4)

    def test_url_validation_and_normalization(self) -> None:
        self.assertEqual(
            normalize_website_url("HTTPS://Example.com/article/#section"),
            "https://example.com/article",
        )
        with self.assertRaises(ValueError):
            validate_website_url("file:///C:/Windows/win.ini")
        with self.assertRaises(ValueError):
            validate_website_url("http://localhost/private")

    def test_scraper_rejects_hostnames_resolving_to_private_addresses(self) -> None:
        private_resolution = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ]
        with (
            patch("scraper.socket.getaddrinfo", return_value=private_resolution),
            patch("scraper.requests.get") as get,
        ):
            with self.assertRaisesRegex(
                WebsiteScrapeError, "Private and reserved IP addresses"
            ):
                scrape_website_with_metadata("https://public-looking.example")

        get.assert_not_called()

    def test_api_key_uses_streamlit_secret_when_environment_is_empty(self) -> None:
        with patch("api_keys.load_dotenv"):
            with patch.dict("os.environ", {}, clear=True):
                with patch(
                    "api_keys.st",
                    SimpleNamespace(
                        secrets={"GROQ_API_KEY": "cloud-test-key"}
                    ),
                ):
                    self.assertEqual(get_api_key("GROQ_API_KEY"), "cloud-test-key")

    def test_missing_tavily_key_has_a_configuration_message(self) -> None:
        with patch("api_keys.load_dotenv"):
            with patch("api_keys._get_streamlit_secret", return_value=None):
                with patch.dict("os.environ", {}, clear=True):
                    with self.assertRaisesRegex(
                        RuntimeError, "TAVILY_API_KEY is missing"
                    ):
                        search_web("a test topic")

    def test_empty_knowledge_base_loads_when_both_files_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_path = Path(temporary_directory) / "missing-index.bin"
            chunks_path = Path(temporary_directory) / "missing-chunks.json"
            index, chunks = load_vector_store(index_path, chunks_path)

        self.assertEqual(index.ntotal, 0)
        self.assertEqual(index.d, 384)
        self.assertEqual(chunks, [])

    def test_incomplete_knowledge_base_is_not_treated_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_path = Path(temporary_directory) / "index.bin"
            chunks_path = Path(temporary_directory) / "chunks.json"
            faiss_index = create_vector_store(
                np.asarray([[1.0, 0.0]], dtype=np.float32)
            )
            faiss.write_index(faiss_index, str(index_path))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_vector_store(index_path, chunks_path)

    def test_corrupted_chunk_json_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_path = Path(temporary_directory) / "index.bin"
            chunks_path = Path(temporary_directory) / "chunks.json"
            index = create_vector_store(np.zeros((1, 384), dtype=np.float32))
            faiss.write_index(index, str(index_path))
            chunks_path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unreadable or corrupted"):
                load_vector_store(index_path, chunks_path)

    def test_embedding_model_download_failure_has_a_friendly_error(self) -> None:
        get_embedding_model.cache_clear()
        try:
            with patch("embeddings.SentenceTransformer", side_effect=OSError):
                with self.assertRaisesRegex(
                    EmbeddingModelError, "model files may need to be downloaded"
                ):
                    get_embedding_model()
        finally:
            get_embedding_model.cache_clear()

    def test_prompt_contains_source_citation_and_question(self) -> None:
        prompt = build_rag_prompt(
            "What is the key idea?",
            [
                {
                    "title": "Research page",
                    "url": "https://example.com/research",
                    "text": "The key idea is grounded retrieval.",
                }
            ],
        )
        self.assertIn("[Source 1: Research page]", prompt)
        self.assertIn("The key idea is grounded retrieval.", prompt)
        self.assertIn("What is the key idea?", prompt)

    def test_prompt_supports_general_chat_without_retrieved_sources(self) -> None:
        prompt = build_rag_prompt("Can you explain photosynthesis?", [])
        self.assertIn("general-purpose conversational assistant", prompt)
        self.assertIn("No relevant external source excerpts", prompt)
        self.assertIn("Can you explain photosynthesis?", prompt)

    def test_prompt_labels_uploaded_file_sources(self) -> None:
        prompt = build_rag_prompt(
            "What does this report say?",
            [
                {
                    "title": "report.pdf",
                    "url": "uploaded://hash/report.pdf",
                    "file_name": "report.pdf",
                    "text": "The report's main finding is improved reliability.",
                }
            ],
        )

        self.assertIn("[Source 1: Uploaded file - report.pdf]", prompt)
        self.assertIn("File: report.pdf", prompt)
        self.assertIn("answer questions about that file from its excerpts first", prompt)

    def test_prompt_keeps_uploaded_file_ahead_of_external_context(self) -> None:
        prompt = build_rag_prompt(
            "What does the report say?",
            [
                {
                    "title": "report.pdf",
                    "url": "uploaded://hash/report.pdf",
                    "file_name": "report.pdf",
                    "text": "The report recommends a 12-month study.",
                },
                {
                    "title": "External article",
                    "url": "https://example.com/article",
                    "text": "External background information.",
                },
            ],
        )

        self.assertLess(
            prompt.index("[Source 1: Uploaded file - report.pdf]"),
            prompt.index("[Source 2: External article]"),
        )

    def test_search_query_uses_context_for_follow_up_questions(self) -> None:
        query = build_search_query(
            "What about in India?",
            [
                {"role": "user", "content": "How does solar power work?"},
                {"role": "assistant", "content": "Solar panels convert sunlight."},
            ],
        )
        self.assertEqual(query, "How does solar power work? What about in India?")
        self.assertLessEqual(len(query), 300)

    def test_appending_preserves_existing_index_records(self) -> None:
        existing_chunks = [
            {"text": "existing", "url": "https://existing.example", "title": "Existing"}
        ]
        new_chunks = [
            {"text": "new", "url": "https://new.example", "title": "New"}
        ]
        existing_embedding = np.zeros((1, 384), dtype=np.float32)
        existing_embedding[0, 0] = 1.0
        new_embedding = np.zeros((1, 384), dtype=np.float32)
        new_embedding[0, 1] = 1.0
        index = create_vector_store(existing_embedding)

        with tempfile.TemporaryDirectory() as temporary_directory:
            index_path = Path(temporary_directory) / "index.bin"
            chunks_path = Path(temporary_directory) / "chunks.json"
            updated_index, updated_chunks = append_to_vector_store(
                index,
                existing_chunks,
                new_chunks,
                new_embedding,
                index_path,
                chunks_path,
            )
            saved_index, saved_chunks = load_vector_store(index_path, chunks_path)

        self.assertEqual(updated_index.ntotal, 2)
        self.assertEqual(saved_index.ntotal, 2)
        self.assertEqual(len(updated_chunks), 2)
        self.assertEqual(saved_chunks[0], existing_chunks[0])
        self.assertEqual(saved_chunks[1], new_chunks[0])

    def test_first_index_append_works_with_empty_knowledge_base(self) -> None:
        index = faiss.IndexFlatL2(384)
        new_chunks = [
            {"text": "first indexed text", "url": "https://first.example"}
        ]
        vectors = np.zeros((1, 384), dtype=np.float32)
        vectors[0, 0] = 1.0
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_path = Path(temporary_directory) / "index.bin"
            chunks_path = Path(temporary_directory) / "chunks.json"
            updated_index, updated_chunks = append_to_vector_store(
                index,
                [],
                new_chunks,
                vectors,
                index_path,
                chunks_path,
            )
            saved_index, saved_chunks = load_vector_store(index_path, chunks_path)

        self.assertEqual(updated_index.ntotal, 1)
        self.assertEqual(saved_index.ntotal, 1)
        self.assertEqual(updated_chunks, new_chunks)
        self.assertEqual(saved_chunks, new_chunks)

    def test_existing_duplicate_url_is_not_scraped(self) -> None:
        chunks = [
            {"text": "existing", "url": "https://example.com/", "title": "Example"}
        ]
        index = create_vector_store(np.asarray([[1.0, 0.0]], dtype=np.float32))
        with patch("rag_pipeline.scrape_website_with_metadata") as scrape:
            _, _, report = update_knowledge_base(
                ["https://EXAMPLE.com/#top"],
                index,
                chunks,
            )
        scrape.assert_not_called()
        self.assertEqual(report["skipped_urls"], ["https://example.com/"])

    def test_duplicate_urls_in_one_update_are_scraped_once(self) -> None:
        index = create_vector_store(np.asarray([[1.0, 0.0]], dtype=np.float32))
        page = {
            "url": "https://new.example/article",
            "title": "New page",
            "content": "Fresh research text.",
        }
        with (
            patch("rag_pipeline.scrape_website_with_metadata", return_value=page) as scrape,
            patch(
                "rag_pipeline.generate_embeddings",
                return_value=np.asarray([[0.0, 1.0]], dtype=np.float32),
            ),
            patch("rag_pipeline.append_to_vector_store") as append,
        ):
            def append_chunks(old_index, old_chunks, new_chunks, vectors):
                self.assertEqual(len(new_chunks), len(vectors))
                return old_index, old_chunks + new_chunks

            append.side_effect = append_chunks
            _, updated_chunks, report = update_knowledge_base(
                ["https://new.example/article", "https://NEW.example/article/"],
                index,
                [],
            )

        scrape.assert_called_once()
        self.assertEqual(len(updated_chunks), 1)
        self.assertEqual(report["new_chunks"], 1)
        self.assertEqual(report["added_urls"], ["https://new.example/article"])

    def test_tavily_uses_bearer_authentication(self) -> None:
        response = Mock()
        response.json.return_value = {
            "results": [
                {
                    "url": "https://research.example/article",
                    "title": "Research",
                    "score": 0.9,
                }
            ]
        }
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
            with patch("search.requests.post", return_value=response) as post:
                results = search_web("a test topic")

        self.assertEqual(results[0]["url"], "https://research.example/article")
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-key",
        )

    def test_tavily_requests_fast_search_snippets(self) -> None:
        response = Mock()
        response.json.return_value = {"results": []}
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
            with patch("search.requests.post", return_value=response) as post:
                search_web("a test topic")

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["search_depth"], "fast")
        self.assertEqual(payload["chunks_per_source"], 3)

    def test_tavily_rate_limit_has_a_friendly_error(self) -> None:
        response = Mock()
        response.status_code = 429
        response.raise_for_status.side_effect = requests.HTTPError(
            "sensitive provider detail", response=response
        )
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
            with patch("search.requests.post", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "request limit was reached"):
                    search_web("a test topic")

    def test_groq_uses_chat_completions_endpoint(self) -> None:
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Grounded answer."}}]
        }
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch("groq_client.requests.post", return_value=response) as post:
                answer = generate_answer("short prompt")

        self.assertEqual(answer, "Grounded answer.")
        post.assert_called_once()
        self.assertEqual(
            post.call_args.args[0],
            "https://api.groq.com/openai/v1/chat/completions",
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-key",
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["model"],
            "openai/gpt-oss-20b",
        )
        self.assertEqual(post.call_args.kwargs["json"]["max_completion_tokens"], 768)
        self.assertEqual(post.call_args.kwargs["timeout"], 30)

    def test_groq_receives_prior_conversation_turns(self) -> None:
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "It works in India too."}}]
        }
        history = [
            {"role": "user", "content": "How does solar power work?"},
            {"role": "assistant", "content": "Solar panels convert sunlight."},
        ]
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch("groq_client.requests.post", return_value=response) as post:
                generate_answer("What about in India?", history)

        messages = post.call_args.kwargs["json"]["messages"]
        self.assertEqual(
            [(message["role"], message["content"]) for message in messages[1:]],
            [
                ("user", "How does solar power work?"),
                ("assistant", "Solar panels convert sunlight."),
                ("user", "What about in India?"),
            ],
        )

    def test_groq_timeout_becomes_a_friendly_error(self) -> None:
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch(
                "groq_client.requests.post",
                side_effect=requests.Timeout,
            ):
                with self.assertRaisesRegex(
                    GroqClientError, "did not respond within 30 seconds"
                ):
                    generate_answer("short prompt")

    def test_groq_server_error_is_reported_without_raw_details(self) -> None:
        response = Mock()
        response.status_code = 504
        response.raise_for_status.side_effect = requests.HTTPError(
            "internal gateway detail", response=response
        )
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch("groq_client.requests.post", return_value=response):
                with self.assertRaisesRegex(GroqClientError, "server error \\(504\\)"):
                    generate_answer("short prompt")

    def test_groq_auth_model_and_rate_limit_errors_are_friendly(self) -> None:
        cases = [
            (401, "Groq rejected this API key"),
            (403, "Groq rejected this API key"),
            (404, "could not find model"),
            (429, "request limit was reached"),
        ]
        for status_code, expected_message in cases:
            with self.subTest(status_code=status_code):
                response = Mock()
                response.status_code = status_code
                response.raise_for_status.side_effect = requests.HTTPError(
                    "private provider detail", response=response
                )
                with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                    with patch("groq_client.requests.post", return_value=response):
                        with self.assertRaisesRegex(
                            GroqClientError, expected_message
                        ):
                            generate_answer("short prompt")

    def test_groq_key_is_required_even_if_gemini_key_exists(self) -> None:
        with patch("api_keys.load_dotenv"):
            with patch("api_keys._get_streamlit_secret", return_value=None):
                with patch.dict(
                    "os.environ", {"GEMINI_API_KEY": "old-key"}, clear=True
                ):
                    self.assertFalse(groq_api_key_configured())
                    with self.assertRaisesRegex(
                        GroqClientError, "cannot be used with Groq"
                    ):
                        generate_answer("short prompt")


if __name__ == "__main__":
    unittest.main()
