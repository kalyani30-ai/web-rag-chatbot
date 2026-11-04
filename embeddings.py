"""Load and reuse the project's Sentence Transformer embedding model."""

from functools import lru_cache

from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
import requests
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModelError(RuntimeError):
    """The local sentence embedding model could not be loaded."""


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it across calls."""
    try:
        return SentenceTransformer(MODEL_NAME)
    except (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        ImportError,
        OSError,
        requests.RequestException,
        RuntimeError,
        ValueError,
    ) as error:
        raise EmbeddingModelError(
            "The embedding model could not be loaded. Check your internet "
            "connection and try again; the model files may need to be downloaded."
        ) from error
