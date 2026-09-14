import httpx
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from app.domains.crawler.ssrf import validate_url_safe, SSRFViolation
from typing import Optional

class FetchResult:
    def __init__(self, url: str, status_code: int, html: Optional[str] = None, error: Optional[str] = None):
        self.url = url
        self.status_code = status_code
        self.html = html
        self.error = error

_ROBOTS_CACHE = {}

async def _can_fetch_robots(client: httpx.AsyncClient, url: str, user_agent: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    
    if robots_url in _ROBOTS_CACHE:
        rp = _ROBOTS_CACHE[robots_url]
        return rp.can_fetch(user_agent, url)

    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        resp = await client.get(robots_url, timeout=5.0)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
    except Exception:
        pass
        
    _ROBOTS_CACHE[robots_url] = rp
    return rp.can_fetch(user_agent, url)

async def fetch_url(url: str, respect_robots: bool = True) -> FetchResult:
    user_agent = "NEXUSBot/1.0"
    
    try:
        validate_url_safe(url)
    except SSRFViolation as e:
        return FetchResult(url=url, status_code=0, error=f"SSRF blocked: {str(e)}")

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, max_redirects=3) as client:
        if respect_robots:
            can_fetch = await _can_fetch_robots(client, url, user_agent)
            if not can_fetch:
                return FetchResult(url=url, status_code=403, error="Blocked by robots.txt")
                
        try:
            resp = await client.get(url, headers={"User-Agent": user_agent})
            validate_url_safe(str(resp.url))
            
            if "text/html" not in resp.headers.get("Content-Type", ""):
                return FetchResult(url=str(resp.url), status_code=resp.status_code, error="Not HTML")
                
            return FetchResult(url=str(resp.url), status_code=resp.status_code, html=resp.text)
        except Exception as e:
            return FetchResult(url=url, status_code=0, error=str(e))
