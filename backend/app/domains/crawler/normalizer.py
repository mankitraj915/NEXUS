import urllib.parse

def normalize_url(url: str) -> str:
    """
    Normalizes a URL to a canonical format.
    - Lowers scheme and netloc
    - Removes default ports (80 for http, 443 for https)
    - Removes fragments
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return url

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    if scheme == 'http' and netloc.endswith(':80'):
        netloc = netloc[:-3]
    elif scheme == 'https' and netloc.endswith(':443'):
        netloc = netloc[:-4]

    path = parsed.path
    if not path:
        path = "/"

    normalized = urllib.parse.urlunparse((
        scheme,
        netloc,
        path,
        parsed.params,
        parsed.query,
        "" # Drop fragment
    ))
    return normalized

def resolve_url(base_url: str, link: str) -> str:
    """Resolves a relative link against a base URL."""
    resolved = urllib.parse.urljoin(base_url, link)
    return normalize_url(resolved)
