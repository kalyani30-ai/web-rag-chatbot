"""Load and reuse the project's Sentence Transformer embedding model."""

import logging
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TYPE_CHECKING

from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
import numpy as np
import requests
import streamlit as st

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
logger = logging.getLogger(__name__)


class EmbeddingModelError(RuntimeError):
    """The local sentence embedding model could not be loaded."""


def _load_sentence_transformer() -> "SentenceTransformer":
    """Import the ML stack and construct the model only when first requested."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s.", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    model.eval()
    return model


@st.cache_resource
def get_embedding_model() -> "SentenceTransformer":
    """Load the embedding model on demand and share it across app sessions."""
    try:
        return _load_sentence_transformer()
    except (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        ImportError,
        OSError,
        requests.RequestException,
        RuntimeError,
        ValueError,
    ) as error:
        logger.exception("Could not load embedding model %s.", MODEL_NAME)
        raise EmbeddingModelError(
            "The embedding model could not be loaded. Check your internet "
            "connection and try again; the model files may need to be downloaded."
        ) from error


def iter_embedding_batches(
    texts: Iterable[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> Iterator[np.ndarray]:
    """Yield contiguous embedding batches without retaining the full result matrix."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    text_iterator = iter(texts)
    batch = list(islice(text_iterator, batch_size))
    if not batch:
        return

    model = get_embedding_model()
    while batch:
        vectors = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(batch):
            raise ValueError("The embedding model returned an invalid embedding batch.")
        yield np.ascontiguousarray(vectors, dtype=np.float32)
        batch = list(islice(text_iterator, batch_size))
