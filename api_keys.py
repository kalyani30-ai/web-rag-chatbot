"""Read API keys from local environment variables or Streamlit secrets."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError


_ENV_PATH = Path(__file__).with_name(".env")


def _get_streamlit_secret(name: str) -> str | None:
    """Read one Cloud secret, returning None when no secrets file is configured."""
    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def get_api_key(name: str) -> str | None:
    """Return a configured key, preferring environment variables over secrets."""
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
    value = os.getenv(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _get_streamlit_secret(name)
