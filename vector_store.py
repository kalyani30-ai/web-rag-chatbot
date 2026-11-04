"""Create, search, update, and persist the local FAISS knowledge base."""

import json
import os
from pathlib import Path

import faiss
import numpy as np


INDEX_PATH = Path(__file__).with_name("faiss_index.bin")
CHUNKS_PATH = Path(__file__).with_name("chunks.json")
EMBEDDING_DIMENSION = 384


def create_vector_store(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create an L2 FAISS index from a batch of text embeddings."""
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("At least one two-dimensional embedding is required.")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def search_vector_store(
    index: faiss.Index, query_embedding: np.ndarray, top_k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Search the index for up to ``top_k`` closest vectors."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if index.ntotal == 0:
        raise ValueError("The knowledge base is empty.")

    query = np.ascontiguousarray(
        np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    )
    if query.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension {query.shape[1]} does not match index dimension {index.d}."
        )
    return index.search(query, min(top_k, index.ntotal))


def save_vector_store(
    index: faiss.Index,
    chunks: list[dict[str, str]],
    index_path: str | Path = INDEX_PATH,
    chunks_path: str | Path = CHUNKS_PATH,
) -> None:
    """Write the FAISS index and its matching source chunks to disk."""
    if index.ntotal != len(chunks):
        raise ValueError(
            f"Cannot save mismatched data: {index.ntotal} vectors and {len(chunks)} chunks."
        )

    index_file = Path(index_path)
    chunks_file = Path(chunks_path)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = index_file.with_name(index_file.name + ".tmp")
    temporary_chunks = chunks_file.with_name(chunks_file.name + ".tmp")

    try:
        faiss.write_index(index, str(temporary_index))
        temporary_chunks.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_index, index_file)
        os.replace(temporary_chunks, chunks_file)
    finally:
        temporary_index.unlink(missing_ok=True)
        temporary_chunks.unlink(missing_ok=True)


def append_to_vector_store(
    index: faiss.Index,
    chunks: list[dict[str, str]],
    new_chunks: list[dict[str, str]],
    new_embeddings: np.ndarray,
    index_path: str | Path = INDEX_PATH,
    chunks_path: str | Path = CHUNKS_PATH,
) -> tuple[faiss.Index, list[dict[str, str]]]:
    """Append new vectors without rebuilding existing data, then persist both files."""
    if not new_chunks:
        return index, chunks
    if len(new_chunks) != len(new_embeddings):
        raise ValueError("Each new text chunk must have exactly one embedding.")

    vectors = np.ascontiguousarray(new_embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != index.d:
        raise ValueError(
            f"New embeddings must have {index.d} dimensions to match the saved index."
        )

    # Clone first so a failed disk write cannot leave the cached index half-updated.
    updated_index = faiss.clone_index(index)
    updated_index.add(vectors)
    updated_chunks = [*chunks, *new_chunks]
    save_vector_store(updated_index, updated_chunks, index_path, chunks_path)
    return updated_index, updated_chunks


def load_vector_store(
    index_path: str | Path = INDEX_PATH,
    chunks_path: str | Path = CHUNKS_PATH,
) -> tuple[faiss.Index, list[dict[str, str]]]:
    """Load saved data, or return an empty index if neither file exists."""
    index_file = Path(index_path)
    chunks_file = Path(chunks_path)
    index_exists = index_file.is_file()
    chunks_exist = chunks_file.is_file()
    if not index_exists and not chunks_exist:
        return faiss.IndexFlatL2(EMBEDDING_DIMENSION), []
    if index_exists != chunks_exist:
        raise ValueError(
            "The saved knowledge base is incomplete: both FAISS and chunk files "
            "must be present."
        )

    try:
        index = faiss.read_index(str(index_file))
        with chunks_file.open("r", encoding="utf-8") as file:
            chunks = json.load(file)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            "The saved knowledge base files are unreadable or corrupted."
        ) from error

    if (
        not isinstance(chunks, list)
        or any(
            not isinstance(chunk, dict)
            or not isinstance(chunk.get("text"), str)
            or not isinstance(chunk.get("url"), str)
            for chunk in chunks
        )
        or index.ntotal != len(chunks)
        or index.d != EMBEDDING_DIMENSION
    ):
        raise ValueError(
            "The saved FAISS index and chunks.json are mismatched or use an "
            "incompatible embedding dimension."
        )
    return index, chunks
