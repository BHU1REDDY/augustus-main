import re
from urllib.parse import urlparse, parse_qs, unquote

_YT_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')

def get_youtube_video_id(url_or_id: str) -> str | None:
    """
    Return the YouTube video ID from a URL or bare ID string.
    Supports:
      - https://www.youtube.com/watch?v=VIDEOID
      - https://youtu.be/VIDEOID
      - https://www.youtube.com/embed/VIDEOID
      - https://www.youtube.com/v/VIDEOID
      - https://www.youtube.com/shorts/VIDEOID
      - Mobile/music subdomains, extra params (&t=, &list=, &si=...), etc.
    Returns None if no valid 11-char ID is found.
    """
    if not url_or_id:
        return None
    s = url_or_id.strip()

    # If it's already an 11-char ID
    if _YT_ID_RE.fullmatch(s):
        return s

    # Parse URL
    try:
        parsed = urlparse(s)
    except Exception:
        return None

    # Some share links wrap another URL (e.g., attribution_link with u=)
    qs = parse_qs(parsed.query)
    if "u" in qs:
        wrapped = unquote(qs["u"][0])
        maybe = get_youtube_video_id(wrapped)
        if maybe:
            return maybe

    host = (parsed.netloc or "").lower()

    # youtu.be/<id>
    if host.endswith("youtu.be"):
        # path like "/VIDEOID" or "/VIDEOID/..."
        parts = [p for p in parsed.path.split("/") if p]
        if parts and _YT_ID_RE.fullmatch(parts[0]):
            return parts[0]

    # *.youtube.com paths
    if "youtube.com" in host or "youtube-nocookie.com" in host:
        path = parsed.path or ""
        parts = [p for p in path.split("/") if p]

        # /watch?v=<id>
        if path == "/watch":
            vid = qs.get("v", [None])[0]
            if vid and _YT_ID_RE.fullmatch(vid):
                return vid

        # /embed/<id>, /v/<id>, /shorts/<id>, /live/<id>
        if parts:
            if parts[0] in {"embed", "v", "shorts", "live"} and len(parts) >= 2:
                candidate = parts[1]
                # strip any trailing parameters accidentally included
                candidate = candidate.split("?")[0]
                if _YT_ID_RE.fullmatch(candidate):
                    return candidate

        # /playlist?list=... has no single video id; sometimes also has v=
        vid = qs.get("v", [None])[0]
        if vid and _YT_ID_RE.fullmatch(vid):
            return vid

    return None

# url = input("enter youtube link: ")
# print(get_youtube_video_id(url))