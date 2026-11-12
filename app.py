"""Streamlit interface for the local web-grounded research assistant."""

import logging
import sqlite3
from hashlib import sha256
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("Starting Web RAG Assistant.")

import faiss
import numpy as np
import streamlit as st

from chat_history import (
    clear_conversation,
    create_conversation,
    delete_conversation,
    list_conversations,
    load_messages,
    rename_conversation,
    save_message,
)
from embeddings import (
    EMBEDDING_BATCH_SIZE,
    get_embedding_model,
    iter_embedding_batches,
)
from groq_client import (
    CONNECTION_TEST_PROMPT,
    GroqClientError,
    generate_answer,
    groq_api_key_configured,
)
from prompt_builder import build_rag_prompt
from rag_pipeline import retrieve_relevant_chunks, update_knowledge_base
from search import build_search_query, search_web
from uploaded_documents import (
    DocumentUploadError,
    SUPPORTED_EXTENSIONS,
    extract_uploaded_content,
)
from processor import clean_text, split_into_chunks
from vector_store import EMBEDDING_DIMENSION, create_vector_store, load_vector_store


logger.info("Application modules imported.")

st.set_page_config(
    page_title="Web RAG Assistant",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark Mode"

# Keep the interface styling local to Streamlit so no extra frontend package is needed.
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --ink: #e8f0fb;
        --muted: #9aacc3;
        --line: rgba(154, 183, 219, .14);
    }
    .stApp {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 70% -15%, rgba(35,91,128,.30), transparent 42%),
            radial-gradient(ellipse at 15% 50%, rgba(24,78,82,.13), transparent 45%),
            #0a1220;
        color-scheme: dark;
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: transparent;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px; padding-top: 5rem; padding-bottom: 7rem;
    }
    [data-testid="stSidebarContent"] { padding-top: 1.25rem; }
    [data-testid="stHeader"] {
        background: rgba(10,18,32,.78);
        border-bottom: 1px solid rgba(154,183,219,.08);
    }
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"], [data-testid="stWidgetLabel"],
    [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    [data-testid="stMarkdownContainer"] strong { color: var(--ink); }
    [data-testid="stSidebar"] {
        color: var(--ink);
        background: linear-gradient(180deg,#101c2e 0%,#0c1727 55%,#0a1422 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #edf5ff; letter-spacing: -.025em; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .75rem; }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        padding: .8rem .9rem; border: 1px solid var(--line);
        border-radius: 14px; background: rgba(255,255,255,.035);
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #e8f5ff; font-weight: 700;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #91a5bd; }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255,255,255,.035);
        border: 1px solid var(--line); border-radius: 14px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary { color: #dce8f7; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        color: var(--ink); background: #111f33;
        border-color: rgba(154,183,219,.20);
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: var(--ink); }
    [data-testid="stSidebar"] [data-baseweb="popover"] {
        background: #142239; border: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [role="option"] { color: var(--ink); }
    [data-testid="stSidebar"] hr { border-color: rgba(154,183,219,.13); }
    .hero {
        position: relative; overflow: hidden;
        padding: 1.35rem 1.8rem; border-radius: 22px; margin: .25rem 0 1.2rem;
        color: #f5fbff;
        background:
            radial-gradient(circle at 88% 18%,rgba(101,225,203,.20),transparent 26%),
            linear-gradient(120deg,#15243b 0%,#173b55 56%,#14515b 100%);
        border: 1px solid rgba(155,221,226,.22);
        box-shadow: 0 22px 65px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.08);
    }
    .hero::after {
        content: ""; position: absolute; width: 300px; height: 300px;
        right: -100px; bottom: -225px; border-radius: 50%;
        border: 1px solid rgba(142,235,219,.18);
        box-shadow: 0 0 0 28px rgba(142,235,219,.025), 0 0 0 58px rgba(142,235,219,.02);
        animation: heroDrift 16s ease-in-out infinite alternate;
    }
    @keyframes heroDrift {
        from { transform: translate3d(0,0,0) scale(1); }
        to { transform: translate3d(-18px,-12px,0) scale(1.05); }
    }
    .hero-kicker {
        margin-bottom: .45rem; color: #9ce8d9; font-size: .68rem;
        font-weight: 750; letter-spacing: .16em; text-transform: uppercase;
    }
    .hero h1 {
        margin: 0 0 .38rem; font-size: clamp(1.7rem,3vw,2.3rem);
        letter-spacing: -.045em; color: #f4f9ff;
    }
    .hero p { margin: 0; color: #bfd0e1; font-size: 1rem; }
    .hero-pills { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .8rem; }
    .hero-pills span {
        padding: .28rem .55rem; border: 1px solid rgba(195,229,237,.18);
        border-radius: 999px; background: rgba(8,20,34,.22);
        color: #d6e8f1; font-size: .7rem;
    }
    .welcome-card {
        position: relative; padding: 1.25rem 1.45rem 1.25rem 1.6rem;
        border: 1px solid var(--line); border-radius: 18px;
        background: linear-gradient(120deg,rgba(20,36,57,.92),rgba(15,29,46,.84));
        color: var(--ink); margin: .4rem 0 1.25rem;
        box-shadow: 0 12px 32px rgba(0,0,0,.12);
    }
    .welcome-card::before {
        content: ""; position: absolute; left: 0; top: 18px; bottom: 18px;
        width: 3px; border-radius: 4px; background: linear-gradient(#68dcc4,#6eabff);
    }
    div[data-testid="stChatMessage"] {
        background: linear-gradient(145deg,rgba(19,32,51,.94),rgba(14,25,41,.92));
        border: 1px solid var(--line); border-radius: 18px;
        padding: .85rem 1.05rem; color: var(--ink);
        box-shadow: 0 8px 26px rgba(0,0,0,.12);
    }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        color: #dce7f3; line-height: 1.72;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg,rgba(24,49,68,.96),rgba(18,39,56,.94));
        border-color: rgba(96,216,195,.17);
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(145deg,rgba(19,32,51,.96),rgba(14,25,41,.94));
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        background: rgba(255,255,255,.035);
        border: 1px solid var(--line); border-radius: 12px;
    }
    [data-testid="stChatMessage"] a { color: #81dcca; }
    div[data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] .stButton > button {
        min-height: 2rem; padding: .25rem .55rem;
        color: #b7d7dc !important; background: rgba(96,216,195,.055);
        border-color: rgba(96,216,195,.14); border-radius: 999px;
        font-size: .78rem;
    }
    div[data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] .stButton > button:hover {
        color: #eafffa !important; background: rgba(96,216,195,.14);
        border-color: rgba(96,216,195,.38); transform: translateY(-1px);
    }
    [data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--line); }
    div[data-testid="stChatInput"] {
        background: rgba(16,29,47,.96);
        border: 1px solid rgba(132,170,207,.25); border-radius: 20px;
        box-shadow: 0 16px 46px rgba(0,0,0,.30), 0 0 0 1px rgba(255,255,255,.025);
        backdrop-filter: blur(18px);
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: rgba(96,216,195,.62);
        box-shadow: 0 16px 46px rgba(0,0,0,.30), 0 0 0 3px rgba(96,216,195,.10);
    }
    div[data-testid="stChatInput"] textarea {
        background: transparent; color: #e8f0fb; border-color: transparent; font-size: 1rem;
    }
    div[data-testid="stChatInput"] textarea::placeholder { color: #8194aa; }
    div[data-testid="stChatInput"] > div { background: transparent; }
    div[data-testid="stChatInput"] button[aria-label="Upload files"] {
        width: 2.25rem; height: 2.25rem; padding: 0;
        color: #92e8d8 !important; background: rgba(88,188,181,.12);
        border: 1px solid rgba(114,220,203,.30); border-radius: 50%;
    }
    div[data-testid="stChatInput"] button[aria-label="Upload files"]:hover {
        color: #d5fff6 !important; background: rgba(88,188,181,.24);
    }
    div[data-testid="stChatInput"] button[aria-label="Send message"] {
        color: #0b1a24 !important; background: linear-gradient(135deg,#83e6cf,#67cbd0);
        border: 0; border-radius: 12px; box-shadow: 0 5px 16px rgba(83,211,194,.20);
    }
    div[data-testid="stChatInput"] button[aria-label="Send message"]:disabled {
        color: #a4b7c7 !important; background: rgba(100,125,149,.17); box-shadow: none;
    }
    .stButton > button {
        min-height: 2.55rem; color: #dcecff !important;
        background: linear-gradient(135deg,rgba(35,67,91,.94),rgba(25,53,75,.94));
        border: 1px solid rgba(127,174,205,.20); border-radius: 12px;
        transition: transform .16s ease,border-color .16s ease,background .16s ease;
    }
    .stButton > button p { color: inherit !important; }
    .stButton > button:hover {
        color: #f2fffc !important; background: linear-gradient(135deg,#24556a,#1b5962);
        border-color: rgba(108,223,203,.45); transform: translateY(-1px);
    }
    .stButton > button:disabled {
        color: #8496aa !important; background: rgba(76,99,123,.22);
        border-color: rgba(154,183,219,.10);
    }
    [data-testid="stLinkButton"] a,
    [data-testid="stLinkButton"] a:hover {
        color: #dcfff6 !important; background: rgba(42,104,105,.56);
        border: 1px solid rgba(114,220,203,.25); border-radius: 10px;
    }
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
        background: linear-gradient(180deg,rgba(10,18,32,0),rgba(10,18,32,.92) 22%);
    }
    [data-testid="stStatusWidget"] {
        border: 1px solid var(--line); border-radius: 14px; background: rgba(16,29,47,.90);
    }
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg,#60d8c3,#79aaff);
    }
    [data-testid="stFileUploader"] { border-color: rgba(154,183,219,.20); }
    ::selection { color: #07151c; background: #72dfcb; }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 4.5rem .85rem 6rem; }
        .hero { padding: 1.1rem; border-radius: 18px; }
        .hero h1 { font-size: 1.6rem; }
        .hero-pills { gap: .4rem; }
        .hero-pills span { font-size: .7rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

theme_colors = {
    "Dark Mode": {
        "scheme": "dark",
        "main": "#000000",
        "sidebar": "#0A0A0A",
        "user": "#171717",
        "input": "#171717",
        "primary": "#FFFFFF",
        "secondary": "#A3A3A3",
        "border": "#292929",
        "hover": "#242424",
        "button": "#111111",
    },
    "Light Mode": {
        "scheme": "light",
        "main": "#FFFFFF",
        "sidebar": "#F7F7F7",
        "user": "#F1F1F1",
        "input": "#FFFFFF",
        "primary": "#111111",
        "secondary": "#666666",
        "border": "#E5E5E5",
        "hover": "#EAEAEA",
        "button": "#FFFFFF",
    },
}
theme = theme_colors[st.session_state.theme_mode]
st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: {theme["scheme"]};
        --primary-color: {theme["primary"]};
        --background-color: {theme["main"]};
        --secondary-background-color: {theme["sidebar"]};
        --text-color: {theme["primary"]};
        --border-color: {theme["border"]};
        --app-bg: {theme["main"]};
        --sidebar-bg: {theme["sidebar"]};
        --user-bg: {theme["user"]};
        --input-bg: {theme["input"]};
        --primary: {theme["primary"]};
        --secondary: {theme["secondary"]};
        --border: {theme["border"]};
        --hover: {theme["hover"]};
        --button-bg: {theme["button"]};
    }}
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
        color: var(--primary) !important;
        background: var(--app-bg) !important;
    }}
    [data-testid="stHeader"], [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {{
        background: var(--app-bg) !important;
        border-color: var(--border) !important;
    }}
    input[type="radio"], input[type="checkbox"] {{
        accent-color: var(--primary) !important;
    }}
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        color: var(--primary) !important;
        background: var(--sidebar-bg) !important;
        border-color: var(--border) !important;
    }}
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"],
    [data-testid="stMetricLabel"], [data-testid="stMarkdownContainer"] li {{
        color: var(--secondary) !important;
    }}
    h1, h2, h3, h4, strong, label,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMetricValue"] {{
        color: var(--primary) !important;
    }}
    [data-testid="stSidebar"] hr {{ border-color: var(--border) !important; }}
    .hero, .welcome-card {{
        color: var(--primary) !important;
        background: var(--app-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }}
    .hero::after, .welcome-card::before, .hero-pills {{ display: none !important; }}
    .hero h1, .hero p, .hero-kicker {{ color: var(--primary) !important; }}
    div[data-testid="stChatMessage"] {{
        color: var(--primary) !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
    }}
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        background: var(--user-bg) !important;
        border-color: var(--border) !important;
        border-radius: 16px !important;
    }}
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {{
        color: var(--primary) !important;
        line-height: 1.7;
    }}
    a, [data-testid="stChatMessage"] a {{
        color: var(--primary) !important;
        text-decoration-color: var(--secondary) !important;
    }}
    input, textarea, [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div, [data-baseweb="select"] > div {{
        color: var(--primary) !important;
        background: var(--input-bg) !important;
        border-color: var(--border) !important;
    }}
    input::placeholder, textarea::placeholder {{
        color: var(--secondary) !important;
    }}
    div[data-testid="stChatInput"] {{
        color: var(--primary) !important;
        background: var(--input-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        box-shadow: none !important;
    }}
    div[data-testid="stChatInput"]:focus-within {{
        border-color: var(--secondary) !important;
        box-shadow: 0 0 0 1px var(--secondary) !important;
    }}
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] > div {{
        color: var(--primary) !important;
        background: transparent !important;
    }}
    .stButton > button, [data-testid="stLinkButton"] a,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stChatInput"] button {{
        color: var(--primary) !important;
        background: var(--button-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }}
    .stButton > button:hover, [data-testid="stLinkButton"] a:hover,
    [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stChatInput"] button:hover {{
        color: var(--primary) !important;
        background: var(--hover) !important;
        border-color: var(--secondary) !important;
        transform: none !important;
    }}
    .stButton > button:disabled,
    [data-testid="stChatInput"] button:disabled {{
        color: var(--secondary) !important;
        background: var(--button-bg) !important;
        opacity: .6;
    }}
    [data-testid="stExpander"], [data-testid="stStatusWidget"],
    [data-testid="stAlert"], [data-testid="stFileUploader"],
    [data-testid="stMetric"], [data-testid="stPopover"],
    [data-testid="stFileUploaderDropzone"],
    [data-baseweb="popover"], [role="dialog"] {{
        color: var(--primary) !important;
        background: var(--sidebar-bg) !important;
        border-color: var(--border) !important;
        box-shadow: none !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stAlert"] *,
    [data-testid="stFileUploader"] *,
    [data-baseweb="popover"] *, [role="dialog"] * {{
        color: var(--primary) !important;
    }}
    [data-testid="stProgressBar"] > div,
    [data-testid="stProgressBar"] > div > div {{
        background: var(--border) !important;
    }}
    [data-testid="stProgressBar"] > div > div > div {{
        background: var(--secondary) !important;
    }}
    [data-testid="stChatInput"] button[aria-label="Upload files"] {{
        background: var(--button-bg) !important;
        border-color: var(--border) !important;
    }}
    div[data-testid="stChatInput"] button[aria-label="Send message"],
    div[data-testid="stChatInput"] button[aria-label="Send message"]:hover,
    div[data-testid="stChatInput"] button[aria-label="Upload files"],
    div[data-testid="stChatInput"] button[aria-label="Upload files"]:hover {{
        color: var(--primary) !important;
        background: var(--button-bg) !important;
        border: 1px solid var(--border) !important;
        box-shadow: none !important;
    }}
    div[data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] .stButton > button,
    div[data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] .stButton > button:hover {{
        color: var(--primary) !important;
        background: var(--button-bg) !important;
        border: 1px solid var(--border) !important;
        box-shadow: none !important;
        transform: none !important;
    }}
    [role="option"], [data-baseweb="popover"] *,
    [data-testid="stDialog"] *, [role="dialog"] * {{
        color: var(--primary) !important;
        background-color: transparent;
    }}
    ::selection {{ color: var(--app-bg); background: var(--secondary); }}
    @media (max-width: 700px) {{
        [data-testid="stMainBlockContainer"] {{ padding: 4.5rem .85rem 6rem; }}
        .hero {{ padding: 1.1rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_database() -> tuple[faiss.Index, list[dict[str, str]]]:
    """Load FAISS and its source text once for the lifetime of the app process."""
    return load_vector_store()


def display_sources(sources: list[dict[str, str]]) -> None:
    """Show answer sources in a compact, expandable section."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for number, source in enumerate(sources, start=1):
            file_name = source.get("file_name")
            if file_name:
                st.markdown(f"**{number}. Uploaded file: {file_name}**")
                continue
            domain = urlparse(source["url"]).netloc or source["url"]
            title = source.get("title") or domain
            st.markdown(f"**{number}. {title}**")
            st.link_button(
                f"Open {domain}",
                source["url"],
                use_container_width=False,
            )


def unique_sources(retrieved_chunks: list[dict[str, str | float]]) -> list[dict[str, str]]:
    """Keep the first relevant chunk for each source URL."""
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for chunk in retrieved_chunks:
        source_url = str(chunk["url"])
        if source_url in seen_urls:
            continue
        seen_urls.add(source_url)
        sources.append(
            {
                "url": source_url,
                "title": str(chunk.get("title", "")),
            }
        )
        if chunk.get("file_name"):
            sources[-1]["file_name"] = str(chunk["file_name"])
    return sources


def index_uploaded_files(uploaded_files) -> tuple[int, list[str]]:
    """Extract and index new uploads into the current Streamlit session."""
    known_files: dict[str, dict[str, str]] = st.session_state.uploaded_files
    pending_chunks: list[dict[str, str]] = []
    added_files: list[tuple[str, dict[str, str]]] = []
    errors: list[str] = []
    seen_digests = set(known_files)

    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()
        file_digest = sha256(content).hexdigest()
        if file_digest in seen_digests:
            continue
        try:
            category, extracted_text = extract_uploaded_content(
                uploaded_file.name,
                content,
            )
            chunks = split_into_chunks(
                clean_text(extracted_text),
                chunk_size=500,
                overlap=50,
            )
            if not chunks:
                raise DocumentUploadError("No searchable text was found.")
        except (DocumentUploadError, OSError, RuntimeError, ValueError) as error:
            errors.append(f"{uploaded_file.name}: {error}")
            continue

        seen_digests.add(file_digest)
        file_metadata = {
            "name": uploaded_file.name,
            "category": category,
        }
        added_files.append((file_digest, file_metadata))
        pending_chunks.extend(
            {
                "url": f"uploaded://{file_digest}/{uploaded_file.name}",
                "title": uploaded_file.name,
                "text": text,
                "file_name": uploaded_file.name,
            }
            for text in chunks
        )
        del content, extracted_text, chunks

    if not pending_chunks:
        return 0, errors

    current_index = st.session_state.uploaded_index
    updated_index = (
        faiss.clone_index(current_index)
        if current_index is not None
        else None
    )
    chunk_offset = 0
    for vectors in iter_embedding_batches(
        (chunk["text"] for chunk in pending_chunks),
        batch_size=EMBEDDING_BATCH_SIZE,
    ):
        expected_count = min(
            EMBEDDING_BATCH_SIZE,
            len(pending_chunks) - chunk_offset,
        )
        if vectors.shape != (expected_count, EMBEDDING_DIMENSION):
            raise ValueError("Uploaded file embeddings have an invalid shape.")
        if updated_index is None:
            updated_index = create_vector_store(vectors)
        else:
            if vectors.shape[1] != updated_index.d:
                raise ValueError("Uploaded file embeddings do not match the saved index.")
            updated_index.add(vectors)
        chunk_offset += expected_count
        del vectors

    if chunk_offset != len(pending_chunks) or updated_index is None:
        raise ValueError("The embedding model returned an incomplete upload batch.")

    st.session_state.uploaded_index = updated_index
    st.session_state.uploaded_chunks.extend(pending_chunks)
    known_files.update(added_files)
    return len(added_files), errors


def run_indexing(
    urls: list[str],
    current_index: faiss.Index,
    current_chunks: list[dict[str, str]],
) -> tuple[faiss.Index, list[dict[str, str]]]:
    """Run one user-requested indexing job and refresh the cached database."""
    progress_bar = st.progress(0)
    status = st.status("Preparing the knowledge-base update...", expanded=True)
    def update_progress(message: str, current: int, total: int) -> None:
        status.update(label=message, state="running")
        progress_bar.progress(min(current / max(total, 1), 0.99))

    try:
        updated_index, updated_chunks, report = update_knowledge_base(
            urls,
            current_index,
            current_chunks,
            progress_callback=update_progress,
        )
        if report["new_chunks"]:
            load_database.clear()
            status.update(
                label=(
                    f"Indexed {len(report['added_urls'])} website(s) and "
                    f"{report['new_chunks']} new text chunk(s)."
                ),
                state="complete",
            )
        else:
            status.update(
                label="No new content was added to the knowledge base.",
                state="complete",
            )
        progress_bar.progress(1.0)
        if report["skipped_urls"]:
            st.info(f"Skipped {len(report['skipped_urls'])} duplicate website(s).")
        for failed_url, reason in report["failed_urls"]:
            st.warning(f"Could not index {failed_url}: {reason}")
        return updated_index, updated_chunks
    except (OSError, RuntimeError, ValueError) as error:
        status.update(label="The knowledge-base update failed.", state="error")
        st.error(str(error))
        return current_index, current_chunks
    finally:
        progress_bar.empty()


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Research workspace</div>
        <h1>Web RAG Assistant</h1>
        <p>Bring your files, the live web, and your questions together.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
startup_status = st.empty()
startup_status.info("Loading saved conversations and the knowledge base...")

try:
    logger.info("Loading saved chat history.")
    conversations = list_conversations()
    conversation_ids = [conversation["id"] for conversation in conversations]
    if not conversations:
        active_conversation_id = create_conversation()
        conversations = list_conversations()
        conversation_ids = [conversation["id"] for conversation in conversations]
        st.session_state.active_conversation_id = active_conversation_id
        st.session_state.messages = []
    elif (
        "active_conversation_id" not in st.session_state
        or st.session_state.active_conversation_id not in conversation_ids
    ):
        active_conversation_id = conversations[0]["id"]
        st.session_state.active_conversation_id = active_conversation_id
        st.session_state.messages = load_messages(active_conversation_id)
    elif "messages" not in st.session_state:
        st.session_state.messages = load_messages(
            st.session_state.active_conversation_id
        )
except (OSError, sqlite3.Error, ValueError) as error:
    logger.exception("Could not initialize saved chat history.")
    st.error(f"Saved chat history could not be loaded: {error}")
    st.stop()

if "groq_status" not in st.session_state:
    st.session_state.groq_status = "Not tested"
if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = ""
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = []
if "uploaded_index" not in st.session_state:
    st.session_state.uploaded_index = None

try:
    logger.info("Loading the saved FAISS knowledge base.")
    index, chunks = load_database()
except (OSError, RuntimeError, ValueError) as error:
    logger.exception("Could not load the saved knowledge base.")
    st.error(f"The saved knowledge base could not be loaded: {error}")
    st.stop()
logger.info(
    "Startup complete: %d conversations and %d saved chunks.",
    len(conversations),
    len(chunks),
)
startup_status.empty()

with st.sidebar:
    st.title("Research workspace")
    st.caption(
        "Chat history and indexed websites use local app storage. "
        "Hosted app restarts may reset this data."
    )
    st.radio(
        "Appearance",
        options=["Dark Mode", "Light Mode"],
        format_func=lambda mode: (
            f"☾ {mode}" if mode == "Dark Mode" else f"☼ {mode}"
        ),
        key="theme_mode",
    )
    conversation_by_id = {
        conversation["id"]: conversation for conversation in conversations
    }
    active_conversation = conversation_by_id[st.session_state.active_conversation_id]
    st.caption(f"Current chat: {active_conversation['title']}")
    new_chat_col, clear_col = st.columns(2)
    if new_chat_col.button("＋ New chat", use_container_width=True):
        try:
            new_conversation_id = create_conversation()
        except sqlite3.Error as error:
            st.error(f"Could not create a new chat: {error}")
            st.stop()
        st.session_state.active_conversation_id = new_conversation_id
        st.session_state.messages = []
        st.rerun()
    if clear_col.button("Clear chat", use_container_width=True):
        try:
            clear_conversation(st.session_state.active_conversation_id)
        except sqlite3.Error as error:
            st.error(f"Could not clear this chat: {error}")
            st.stop()
        st.session_state.messages = []
        st.rerun()

    st.caption("Saved conversations")
    for conversation in conversations:
        conversation_id = conversation["id"]
        row_title, row_menu = st.columns([0.84, 0.16])
        title_marker = "● " if conversation_id == st.session_state.active_conversation_id else ""
        if row_title.button(
            f"{title_marker}{conversation['title']}",
            key=f"open_conversation_{conversation_id}",
            use_container_width=True,
            help=f"Open conversation #{conversation_id}",
        ):
            try:
                st.session_state.messages = load_messages(conversation_id)
            except (OSError, sqlite3.Error, ValueError) as error:
                st.error(f"Could not open this saved chat: {error}")
                st.stop()
            st.session_state.active_conversation_id = conversation_id
            st.session_state.editing_conversation_id = None
            st.rerun()

        if row_menu.button(
            "✎",
            key=f"edit_conversation_{conversation_id}",
            help=f"Rename {conversation['title']}",
            use_container_width=True,
        ):
            st.session_state.editing_conversation_id = conversation_id
            st.session_state[f"rename_conversation_title_{conversation_id}"] = (
                conversation["title"]
            )
            st.rerun()

        if st.session_state.get("editing_conversation_id") == conversation_id:
            rename_input_key = f"rename_conversation_title_{conversation_id}"
            st.text_input(
                "Conversation title",
                key=rename_input_key,
                max_chars=120,
                placeholder="Conversation title",
            )
            save_col, cancel_col = st.columns(2)
            if save_col.button(
                "Save",
                key=f"save_conversation_title_{conversation_id}",
                use_container_width=True,
            ):
                try:
                    rename_conversation(
                        conversation_id,
                        st.session_state[rename_input_key],
                    )
                except (OSError, sqlite3.Error, ValueError) as error:
                    st.error(f"Could not rename this saved chat: {error}")
                else:
                    st.session_state.editing_conversation_id = None
                    st.rerun()
            if cancel_col.button(
                "Cancel",
                key=f"cancel_conversation_title_{conversation_id}",
                use_container_width=True,
            ):
                st.session_state.editing_conversation_id = None
                st.rerun()

    if st.button(
        "Delete selected conversation",
        use_container_width=True,
        key="delete_selected_conversation",
    ):
        st.session_state.pending_delete_conversation_id = (
            st.session_state.active_conversation_id
        )
    pending_delete_id = st.session_state.get("pending_delete_conversation_id")
    if pending_delete_id is not None and pending_delete_id in conversation_by_id:
        pending_title = conversation_by_id[pending_delete_id]["title"]
        st.warning(
            f"Permanently delete **{pending_title}** and all of its messages?"
        )
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button(
            "Delete permanently",
            use_container_width=True,
            key="confirm_delete_conversation",
        ):
            try:
                delete_conversation(pending_delete_id)
                conversations = list_conversations()
                if not conversations:
                    create_conversation()
                    conversations = list_conversations()
                next_conversation_id = conversations[0]["id"]
                st.session_state.active_conversation_id = next_conversation_id
                st.session_state.messages = load_messages(next_conversation_id)
                st.session_state.pending_delete_conversation_id = None
                st.rerun()
            except (OSError, sqlite3.Error, ValueError) as error:
                st.error(f"Could not delete this saved chat: {error}")
                st.stop()
        if cancel_col.button(
            "Keep conversation",
            use_container_width=True,
            key="cancel_delete_conversation",
        ):
            st.session_state.pending_delete_conversation_id = None
            st.rerun()

    st.divider()
    st.subheader("Knowledge base")
    indexed_websites = len({chunk.get("url", "") for chunk in chunks if chunk.get("url")})
    website_col, chunk_col = st.columns(2)
    website_col.metric("Websites", indexed_websites)
    chunk_col.metric("Text chunks", len(chunks))
    st.caption("Embedding model")
    st.code("all-MiniLM-L6-v2", language=None)
    st.caption(f"Retrieval: {index.ntotal} vectors · FAISS")

    with st.expander("Session uploads", expanded=False):
        st.caption(
            "Use the + button beside the chat box to attach documents or media. "
            "PDF, DOCX, TXT, MD, CSV, JSON, LOG, common images, and common video "
            "formats are accepted (20 MB per file)."
        )
        st.caption(
            "Images use local Tesseract OCR. Videos are indexed by filename and "
            "metadata only; their audio and frames are not analyzed."
        )
        if st.session_state.uploaded_files:
            st.caption(
                f"{len(st.session_state.uploaded_files)} uploaded file(s) available "
                "to all chats until this app session ends."
            )
            for file_metadata in st.session_state.uploaded_files.values():
                st.markdown(
                    f"- **{file_metadata['name']}** "
                    f"({file_metadata['category']})"
                )
            if st.button("Remove all session uploads", use_container_width=True):
                st.session_state.uploaded_files = {}
                st.session_state.uploaded_chunks = []
                st.session_state.uploaded_index = None
                st.rerun()
        else:
            st.caption("No files attached in this session.")

    top_k = st.slider(
        "Sources to retrieve",
        min_value=1,
        max_value=8,
        value=4,
        help="More sources add context; fewer sources keep answers focused.",
    )
    search_web_automatically = st.toggle(
        "Search the web automatically",
        value=True,
        help="Search current web pages for each question and include relevant excerpts and citations.",
    )

    st.divider()
    st.subheader("Groq connection")
    if groq_api_key_configured():
        st.caption(f"API key configured · {st.session_state.groq_status}")
        if st.button("Test Groq connection", use_container_width=True):
            try:
                with st.spinner("Checking the Groq API..."):
                    response = generate_answer(CONNECTION_TEST_PROMPT)
                st.session_state.groq_status = "Connected"
                st.success(response)
            except GroqClientError as error:
                st.session_state.groq_status = "Unavailable"
                st.error(str(error))
    else:
        st.session_state.groq_status = "API key missing"
        st.warning(
            "Add GROQ_API_KEY to .env locally or to Streamlit Cloud app secrets."
        )
        st.link_button("Get a Groq API key", "https://console.groq.com/keys")

    st.divider()
    with st.expander("Manage knowledge base", expanded=False):
        st.caption("Index a website directly or run a web search on request.")
        with st.form("add_website_form", clear_on_submit=True):
            website_url = st.text_input(
                "Website URL",
                placeholder="https://example.com/article",
            )
            add_website = st.form_submit_button(
                "Scrape and index website",
                use_container_width=True,
            )
        if add_website:
            index, chunks = run_indexing([website_url], index, chunks)

        st.markdown("**Add pages from web search**")
        with st.form("search_web_form", clear_on_submit=True):
            search_query = st.text_input(
                "Topic to search",
                placeholder="e.g. recent advances in battery storage",
            )
            search_and_index = st.form_submit_button(
                "Search and index results",
                use_container_width=True,
            )
        if search_and_index:
            try:
                with st.spinner("Searching the web..."):
                    results = search_web(search_query)
                result_urls = [
                    str(result["url"])
                    for result in results
                    if isinstance(result.get("url"), str)
                ]
                if not result_urls:
                    st.info("No new websites were found for that search.")
                else:
                    index, chunks = run_indexing(result_urls, index, chunks)
            except (RuntimeError, ValueError) as error:
                st.error(str(error))

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <strong>Your next idea starts here.</strong><br>
            Ask a question, attach a document with the <b>+</b> button, or explore a
            topic. I’ll look across your uploaded files, saved sources, and the web,
            then show where the answer came from.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Try a question**")
    suggestion_columns = st.columns(3)
    example_questions = [
        "What are the main ideas in the indexed sources?",
        "Compare two important concepts from the knowledge base.",
        "What evidence supports the key findings?",
    ]
    for column, example in zip(suggestion_columns, example_questions):
        if column.button(example, use_container_width=True):
            st.session_state.suggested_question = example

latest_assistant_index = next(
    (
        message_index
        for message_index in range(len(st.session_state.messages) - 1, -1, -1)
        if st.session_state.messages[message_index]["role"] == "assistant"
    ),
    None,
)
for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            display_sources(message.get("sources", []))
            if message_index == latest_assistant_index:
                action_columns = st.columns(3)
                if action_columns[0].button(
                    "↗ Explain more",
                    key=f"explain_answer_{message_index}",
                    use_container_width=True,
                ):
                    st.session_state.suggested_question = (
                        "Explain your previous answer in more detail, using a "
                        "simple example where it helps."
                    )
                if action_columns[1].button(
                    "▤ Summarize",
                    key=f"summarize_answer_{message_index}",
                    use_container_width=True,
                ):
                    st.session_state.suggested_question = (
                        "Summarize your previous answer in three concise bullet points."
                    )
                if action_columns[2].button(
                    "✦ Suggest follow-ups",
                    key=f"suggest_followups_{message_index}",
                    use_container_width=True,
                ):
                    st.session_state.suggested_question = (
                        "Suggest three useful follow-up questions based on your "
                        "previous answer. Do not answer the questions yet."
                    )

submission = st.chat_input(
    "Ask me anything...",
    accept_file="multiple",
    file_type=sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
    max_upload_size=20,
)
if submission is None:
    typed_question = ""
    attached_files = []
elif isinstance(submission, str):
    typed_question = submission
    attached_files = []
else:
    typed_question = submission.text
    attached_files = submission.files

if attached_files:
    try:
        with st.spinner("Extracting text and indexing attached files..."):
            added_count, upload_errors = index_uploaded_files(attached_files)
        if added_count:
            st.success(
                f"Added {added_count} file(s) to this session's search index."
            )
        elif not upload_errors:
            st.info("These files have already been added to this session.")
        for upload_error in upload_errors:
            st.warning(upload_error)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not index attached files: {error}")

user_question = typed_question or st.session_state.suggested_question
st.session_state.suggested_question = ""

if user_question:
    question = user_question.strip()
    if not question:
        st.warning("Enter a question before sending.")
    elif len(question) > 2_000:
        st.warning("Please keep questions under 2,000 characters.")
    else:
        try:
            save_message(
                st.session_state.active_conversation_id,
                "user",
                question,
            )
        except sqlite3.Error as error:
            st.error(f"Could not save your question: {error}")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            answer: str | None = None
            sources: list[dict[str, str]] = []
            error_message: str | None = None
            web_warning: str | None = None
            conversation_history = st.session_state.messages[:-1]
            with st.status(
                "Preparing your answer...", expanded=True
            ) as response_status:
                try:
                    web_chunks: list[dict[str, str | float]] = []
                    if search_web_automatically:
                        response_status.update(
                            label="Searching the web for relevant sources...",
                            state="running",
                        )
                        try:
                            results = search_web(
                                build_search_query(question, conversation_history),
                                max_results=5,
                            )
                            for result in results:
                                title = result.get("title")
                                url = result.get("url")
                                content = result.get("content")
                                if (
                                    isinstance(url, str)
                                    and isinstance(content, str)
                                    and content.strip()
                                ):
                                    web_chunks.append(
                                        {
                                            "title": str(title or urlparse(url).netloc),
                                            "url": url,
                                            "text": content.strip(),
                                        }
                                    )
                        except (RuntimeError, ValueError) as error:
                            web_warning = (
                                f"Web search is unavailable ({error}). "
                                "I’ll answer using the saved knowledge base and "
                                "general knowledge."
                            )

                    response_status.update(
                        label="Searching your saved knowledge base...",
                        state="running",
                    )
                    retrieved_chunks: list[dict[str, str | float]] = []
                    uploaded_index = st.session_state.uploaded_index
                    embedding_model = None
                    query_embedding = None
                    if index.ntotal or (
                        uploaded_index is not None and uploaded_index.ntotal
                    ):
                        embedding_model = get_embedding_model()
                        query_embedding = embedding_model.encode(
                            [question],
                            convert_to_numpy=True,
                        )[0]
                    if uploaded_index is not None:
                        retrieved_chunks.extend(
                            retrieve_relevant_chunks(
                                question,
                                uploaded_index,
                                st.session_state.uploaded_chunks,
                                top_k=top_k,
                                embedding_model=embedding_model,
                                query_embedding=query_embedding,
                            )
                        )
                    indexed_chunks = retrieve_relevant_chunks(
                        question,
                        index,
                        chunks,
                        top_k=top_k,
                        embedding_model=embedding_model,
                        query_embedding=query_embedding,
                    )
                    retrieved_chunks.extend(indexed_chunks)
                    retrieved_chunks.extend(web_chunks)
                    prompt = build_rag_prompt(question, retrieved_chunks)
                    response_status.update(
                        label="Generating your answer with Groq...",
                        state="running",
                    )
                    history_for_model: list[dict[str, str]] = []
                    for message in conversation_history:
                        role = message.get("role")
                        content = message.get("content")
                        if role in {"user", "assistant"} and isinstance(content, str):
                            history_content = content
                            if role == "assistant" and message.get("sources"):
                                source_labels = [
                                    f"{source.get('title', source.get('url', ''))}: "
                                    f"{source.get('url', '')}"
                                    for source in message["sources"]
                                    if isinstance(source, dict)
                                ]
                                if source_labels:
                                    history_content += (
                                        "\nPreviously cited sources: "
                                        + "; ".join(source_labels)
                                    )
                            history_for_model.append(
                                {"role": str(role), "content": history_content}
                            )
                    answer = generate_answer(prompt, history_for_model)
                    sources = unique_sources(retrieved_chunks)
                    response_status.update(
                        label="Answer ready.",
                        state="complete",
                        expanded=False,
                    )
                except GroqClientError as error:
                    response_status.update(
                        label="Groq could not complete the request.",
                        state="error",
                    )
                    st.session_state.groq_status = "Unavailable"
                    error_message = str(error)
                except (OSError, RuntimeError, ValueError) as error:
                    response_status.update(
                        label="Could not prepare an answer.",
                        state="error",
                    )
                    error_message = str(error)

            if answer is not None:
                if web_warning:
                    st.warning(web_warning)
                st.markdown(answer)
                display_sources(sources)
                try:
                    save_message(
                        st.session_state.active_conversation_id,
                        "assistant",
                        answer,
                        sources,
                    )
                except sqlite3.Error as error:
                    st.error(f"Your answer could not be saved to chat history: {error}")
                    st.stop()
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            elif error_message is not None:
                st.error(error_message)
                try:
                    save_message(
                        st.session_state.active_conversation_id,
                        "assistant",
                        error_message,
                    )
                except sqlite3.Error as error:
                    st.error(f"Could not save the chat error message: {error}")
                    st.stop()
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
