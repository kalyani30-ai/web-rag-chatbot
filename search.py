"""Search Tavily only when a user explicitly starts a knowledge-base update."""

import logging
from collections.abc import Mapping, Sequence
from urllib.parse import urlparse

import requests

from api_keys import get_api_key
from scraper import normalize_website_url, validate_website_url


logger = logging.getLogger(__name__)
_TAVILY_ENDPOINT = "https://api.tavily.com/search"


def build_search_query(
    question: str,
    conversation_history: Sequence[Mapping[str, object]] = (),
) -> str:
    """Include recent user questions so short follow-ups remain searchable."""
    current_question = question.strip() if isinstance(question, str) else ""
    if not current_question:
        raise ValueError("Enter a search query.")
    if len(current_question) >= 300:
        return current_question[:300]

    previous_questions: list[str] = []
    for message in conversation_history:
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str) and content.strip():
            previous_questions.append(content.strip())
    previous_questions = previous_questions[-2:]
    if not previous_questions:
        return current_question

    available_context = 299 - len(current_question)
    previous_context = " ".join(previous_questions)[-available_context:]
    return f"{previous_context} {current_question}"[-300:]


def search_web(query: str, max_results: int = 5) -> list[dict[str, object]]:
    """Return unique, validated Tavily results with source excerpts."""
    cleaned_query = query.strip() if isinstance(query, str) else ""
    if not cleaned_query:
        raise ValueError("Enter a search query.")
    if len(cleaned_query) > 300:
        raise ValueError("Search queries must be 300 characters or fewer.")
    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10.")

    api_key = get_api_key("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Add it to .env or Streamlit Community Cloud secrets."
        )

    payload = {
        "query": cleaned_query,
        "search_depth": "fast",
        "max_results": max_results,
        "chunks_per_source": 3,
    }
    try:
        response = requests.post(
            _TAVILY_ENDPOINT,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=(5, 20),
        )
        response.raise_for_status()
        data = response.json()
    except requests.Timeout as error:
        logger.warning("Tavily search timed out.")
        raise RuntimeError("Web search timed out. Please try again.") from error
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else 0
        logger.warning("Tavily API returned status %s.", status_code)
        if status_code in (401, 403):
            message = "Tavily rejected the API key. Check TAVILY_API_KEY."
        elif status_code == 429:
            message = "Tavily's request limit was reached. Wait before trying again."
        elif status_code in (500, 502, 503, 504):
            message = "Tavily returned a temporary server error. Please try again."
        else:
            message = f"Tavily returned API error {status_code}."
        raise RuntimeError(message) from error
    except requests.RequestException as error:
        logger.warning("Tavily search failed (%s).", type(error).__name__)
        raise RuntimeError(
            "Web search could not be completed. Check the Tavily API key and connection."
        ) from error
    except ValueError as error:
        raise RuntimeError("Tavily returned an unreadable response.") from error

    if not isinstance(data, dict):
        raise RuntimeError("Tavily returned an unexpected response.")

    results = data.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Tavily returned an unexpected results list.")

    filtered_results: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        score = result.get("score", 0)
        if isinstance(score, (int, float)) and score < 0.5:
            continue
        raw_url = result.get("url")
        if not isinstance(raw_url, str):
            continue
        try:
            clean_url = validate_website_url(raw_url)
            normalized_url = normalize_website_url(clean_url)
        except ValueError:
            continue
        if normalized_url in seen_urls:
            continue
        if "bing.com" in (urlparse(clean_url).hostname or "").lower():
            continue
        seen_urls.add(normalized_url)
        filtered_results.append({**result, "url": clean_url})
        if len(filtered_results) >= max_results:
            break
    return filtered_results


if __name__ == "__main__":
    for number, result in enumerate(search_web("What is artificial intelligence?"), 1):
        print(f"{number}. {result.get('title', 'Untitled')}")
        print(f"   {result['url']}")
