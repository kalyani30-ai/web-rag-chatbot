"""Validate public website URLs and extract readable page text and titles."""

import ipaddress
import logging
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)
_MAX_REDIRECTS = 5


class WebsiteScrapeError(RuntimeError):
    """A website could not be validated or its readable text could not be fetched."""


def _ensure_public_dns_addresses(url: str) -> None:
    """Reject hostnames that resolve to private or otherwise non-public addresses."""
    parsed = urlsplit(url)
    hostname = parsed.hostname
    if hostname is None:
        raise WebsiteScrapeError("The website URL does not contain a valid hostname.")
    try:
        addresses = {
            ipaddress.ip_address(result[4][0])
            for result in socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except (OSError, ValueError) as error:
        raise WebsiteScrapeError(
            "The website hostname could not be safely resolved."
        ) from error
    if not addresses or any(not address.is_global for address in addresses):
        raise WebsiteScrapeError(
            "Private and reserved IP addresses cannot be indexed."
        )


def validate_website_url(url: str) -> str:
    """Accept an HTTP(S) URL with a public-looking hostname and no credentials."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Enter a website URL.")

    candidate = url.strip()
    if len(candidate) > 2048:
        raise ValueError("Website URLs must be shorter than 2,048 characters.")

    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a complete website URL starting with http:// or https://.")
    if parsed.username or parsed.password:
        raise ValueError("Website URLs must not contain a username or password.")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("The website URL contains an invalid port.") from error

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(
        (".localhost", ".local", ".internal")
    ):
        raise ValueError("Local and internal hostnames cannot be indexed.")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("Private and reserved IP addresses cannot be indexed.")

    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


def normalize_website_url(url: str) -> str:
    """Normalize URL casing and trailing slashes for duplicate detection."""
    validated = validate_website_url(url)
    parsed = urlsplit(validated)
    hostname = (parsed.hostname or "").lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    port = parsed.port
    if port and not (
        (parsed.scheme == "http" and port == 80)
        or (parsed.scheme == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme, hostname, path, parsed.query, ""))


def scrape_website_with_metadata(url: str) -> dict[str, str]:
    """Download a page and return its final URL, title, and readable text."""
    current_url = validate_website_url(url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        for redirect_count in range(_MAX_REDIRECTS + 1):
            _ensure_public_dns_addresses(current_url)
            response = requests.get(
                current_url,
                headers=headers,
                timeout=(5, 15),
                allow_redirects=False,
            )
            if response.is_redirect:
                location = response.headers.get("Location")
                if not location or redirect_count == _MAX_REDIRECTS:
                    raise WebsiteScrapeError("The website redirected too many times.")
                current_url = validate_website_url(urljoin(current_url, location))
                continue
            response.raise_for_status()
            break
        else:
            raise WebsiteScrapeError("The website redirected too many times.")
    except requests.Timeout as error:
        logger.warning("Website request timed out for %s.", urlsplit(current_url).hostname)
        raise WebsiteScrapeError("The website did not respond before the timeout.") from error
    except requests.RequestException as error:
        logger.warning(
            "Website request failed for %s (%s).",
            urlsplit(current_url).hostname,
            type(error).__name__,
        )
        raise WebsiteScrapeError(
            "The website could not be downloaded. Check the URL and try again."
        ) from error

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        element.decompose()
    content = soup.get_text(separator=" ", strip=True)
    if not content:
        raise WebsiteScrapeError("No readable page text was found at this URL.")

    return {
        "url": response.url,
        "title": title or (urlsplit(response.url).hostname or response.url),
        "content": content,
    }


def scrape_website(url: str) -> str:
    """Return page text for existing callers, or an empty string on scrape failure."""
    try:
        return scrape_website_with_metadata(url)["content"]
    except WebsiteScrapeError:
        return ""


if __name__ == "__main__":
    example_page = scrape_website_with_metadata("https://example.com")
    print(f"Title: {example_page['title']}")
    print(example_page["content"])
