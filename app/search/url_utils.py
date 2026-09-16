"""
Conservative URL normalization, used only to build a comparison key for
deduplication (see app/search/quality.py). Deliberately small: no
aggressive canonicalization, no reordering of ordinary query parameters,
nothing provider-specific.
"""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# "Obvious" tracking parameters - not an exhaustive list, just the common,
# unambiguous ones. Anything else is left alone.
_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAM_NAMES = {"fbclid", "gclid"}


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered in _TRACKING_PARAM_NAMES or lowered.startswith(
        _TRACKING_PARAM_PREFIXES
    )


def normalize_url(url: str) -> str:
    """Return a normalized comparison key for `url` - never the URL to show
    the user; callers keep the original `url` for display/linking.

    Conservative, deterministic rules only:
    - surrounding whitespace is stripped
    - scheme and host are lowercased (case-insensitive by spec); the path
      is left as-is, since path case can be meaningful
    - a single trailing slash is dropped (an empty/root path normalizes
      the same way, so "https://x.com" == "https://x.com/")
    - obvious tracking query parameters (utm_*, fbclid, gclid) are removed
    - all other query parameters and their order, and the fragment, are
      left untouched
    """
    parts = urlsplit(url.strip())

    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()

    path = parts.path
    if path == "/":
        path = ""
    elif path.endswith("/"):
        path = path[:-1]

    query_pairs = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(name)
    ]
    query = urlencode(query_pairs)

    return urlunsplit((scheme, netloc, path, query, parts.fragment))
