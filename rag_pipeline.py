"""Reusable retrieval and incremental website-indexing workflows."""

import logging
from collections.abc import Callable, Iterable
from typing import TypedDict

import faiss
import numpy as np

from embeddings import get_embedding_model
from processor import clean_text, split_into_chunks
from scraper import (
    WebsiteScrapeError,
    normalize_website_url,
    scrape_website_with_metadata,
    validate_website_url,
)
from search import search_web
from vector_store import append_to_vector_store, search_vector_store


logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, int], None]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_K = 4


class IndexingReport(TypedDict):
    """Summary of website updates shown in the Streamlit interface."""

    added_urls: list[str]
    skipped_urls: list[str]
    failed_urls: list[tuple[str, str]]
    new_chunks: int


def collect_websites(question: str) -> list[str]:
    """Search Tavily for website URLs when an update is explicitly requested."""
    return [
        str(result["url"])
        for result in search_web(question)
        if isinstance(result.get("url"), str)
    ]


def collect_website_content(websites: Iterable[str]) -> list[dict[str, str]]:
    """Fetch text for a list of websites while retaining each source URL."""
    documents: list[dict[str, str]] = []
    for url in websites:
        try:
            page = scrape_website_with_metadata(url)
        except (ValueError, WebsiteScrapeError) as error:
            logger.warning("Skipping website %s: %s", url, error)
            continue
        documents.append(
            {
                "url": page["url"],
                "title": page["title"],
                "content": page["content"],
            }
        )
    return documents


def process_documents(
    documents: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    """Clean each scraped page and attach source metadata to every text chunk."""
    all_chunks: list[dict[str, str]] = []
    for document in documents:
        cleaned_content = clean_text(document.get("content", ""))
        for text_chunk in split_into_chunks(
            cleaned_content,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        ):
            all_chunks.append(
                {
                    "url": document["url"],
                    "title": document.get("title", ""),
                    "text": text_chunk,
                }
            )
    return all_chunks


def generate_embeddings(
    chunks: list[dict[str, str]], show_progress_bar: bool = False
) -> np.ndarray:
    """Encode all new chunks in one batch using the shared model instance."""
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)
    texts = [chunk["text"] for chunk in chunks]
    return get_embedding_model().encode(
        texts,
        batch_size=32,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )


def retrieve_relevant_chunks(
    question: str,
    index: faiss.Index,
    chunks: list[dict[str, str]],
    top_k: int = RETRIEVAL_TOP_K,
    embedding_model=None,
    query_embedding: np.ndarray | None = None,
) -> list[dict[str, str | float]]:
    """Encode a question once and return the closest text chunks with sources."""
    if not question or not question.strip():
        raise ValueError("Enter a question before searching the knowledge base.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if index.ntotal == 0:
        return []

    if query_embedding is None:
        model = embedding_model or get_embedding_model()
        query_embedding = model.encode(
            [question.strip()],
            convert_to_numpy=True,
        )[0]
    distances, positions = search_vector_store(index, query_embedding, top_k=top_k)

    retrieved: list[dict[str, str | float]] = []
    for position, chunk_index in enumerate(positions[0]):
        if chunk_index < 0:
            continue
        chunk = chunks[int(chunk_index)]
        retrieved.append(
            {
                "text": chunk["text"],
                "url": chunk["url"],
                "title": chunk.get("title", ""),
                "distance": float(distances[0][position]),
            }
        )
        if chunk.get("file_name"):
            retrieved[-1]["file_name"] = chunk["file_name"]
    return retrieved


def update_knowledge_base(
    websites: Iterable[str],
    index: faiss.Index,
    chunks: list[dict[str, str]],
    progress_callback: ProgressCallback | None = None,
) -> tuple[faiss.Index, list[dict[str, str]], IndexingReport]:
    """Scrape only new URLs, create their embeddings, and append them to FAISS."""
    website_list = list(websites)
    report: IndexingReport = {
        "added_urls": [],
        "skipped_urls": [],
        "failed_urls": [],
        "new_chunks": 0,
    }
    known_urls: set[str] = set()
    for chunk in chunks:
        try:
            known_urls.add(normalize_website_url(chunk.get("url", "")))
        except ValueError:
            continue

    pending_urls: list[str] = []
    seen_urls: set[str] = set()
    for url in website_list:
        try:
            clean_url = validate_website_url(url)
            normalized_url = normalize_website_url(clean_url)
        except ValueError as error:
            report["failed_urls"].append((url, str(error)))
            continue
        if normalized_url in known_urls or normalized_url in seen_urls:
            report["skipped_urls"].append(clean_url)
            continue
        seen_urls.add(normalized_url)
        pending_urls.append(clean_url)

    new_chunks: list[dict[str, str]] = []
    total = len(pending_urls)
    for number, url in enumerate(pending_urls, start=1):
        if progress_callback:
            progress_callback(
                f"Scraping website {number}/{total}: {url}",
                number - 1,
                total + 1,
            )
        try:
            page = scrape_website_with_metadata(url)
            final_url = page["url"]
            normalized_final_url = normalize_website_url(final_url)
            if normalized_final_url in known_urls:
                report["skipped_urls"].append(final_url)
                continue

            if progress_callback:
                progress_callback(
                    f"Cleaning and chunking website {number}/{total}",
                    number,
                    total + 1,
                )
            cleaned_content = clean_text(page["content"])
            page_chunks = split_into_chunks(
                cleaned_content,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
            if not page_chunks:
                raise WebsiteScrapeError("No readable page text was found at this URL.")
        except (ValueError, WebsiteScrapeError) as error:
            report["failed_urls"].append((url, str(error)))
            if progress_callback:
                progress_callback(
                    f"Finished website {number}/{total}",
                    number,
                    total + 1,
                )
            continue

        known_urls.add(normalized_final_url)
        new_chunks.extend(
            {"url": final_url, "title": page["title"], "text": text}
            for text in page_chunks
        )
        report["added_urls"].append(final_url)
        if progress_callback:
            progress_callback(
                f"Scraped website {number}/{total}",
                number,
                total + 1,
            )

    if not new_chunks:
        if progress_callback:
            progress_callback("No new website content to index", total, total or 1)
        return index, chunks, report

    if progress_callback:
        progress_callback("Generating embeddings", total, total + 1)
    embeddings = generate_embeddings(new_chunks, show_progress_bar=False)
    updated_index, updated_chunks = append_to_vector_store(
        index,
        chunks,
        new_chunks,
        embeddings,
    )
    report["new_chunks"] = len(new_chunks)
    if progress_callback:
        progress_callback("Saving the updated knowledge base", total + 1, total + 1)
    return updated_index, updated_chunks, report


if __name__ == "__main__":
    from vector_store import load_vector_store

    query = input("What topic should be added to the knowledge base? ").strip()
    if not query:
        raise SystemExit("Please enter a search query.")

    current_index, current_chunks = load_vector_store()
    new_urls = collect_websites(query)
    _, _, update_report = update_knowledge_base(
        new_urls,
        current_index,
        current_chunks,
    )
    print(
        f"Added {len(update_report['added_urls'])} websites "
        f"and {update_report['new_chunks']} chunks."
    )
