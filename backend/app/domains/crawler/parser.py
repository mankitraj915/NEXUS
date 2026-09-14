from urllib.parse import urljoin

from bs4 import BeautifulSoup


class ParsedDocument:
    def __init__(
        self, title: str, text: str, links: list[str], metadata: dict[str, str]
    ):
        self.title = title
        self.text = text
        self.links = links
        self.metadata = metadata


def parse_html(html: str, base_url: str) -> ParsedDocument:
    soup = BeautifulSoup(html, "html.parser")

    for script_or_style in soup(["script", "style", "noscript", "iframe"]):
        script_or_style.extract()

    title_tag = soup.title
    title = title_tag.string.strip() if title_tag and title_tag.string else ""

    text = soup.get_text(separator="\n", strip=True)

    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href_raw = a["href"]
        href_str = (
            href_raw if isinstance(href_raw, str) else href_raw[0] if href_raw else ""
        )
        if href_str.startswith(("http", "https", "/")):
            links.append(urljoin(base_url, href_str))

    metadata: dict[str, str] = {}
    for meta in soup.find_all("meta"):
        key = meta.get("name") or meta.get("property")
        if key:
            key_str = key if isinstance(key, str) else key[0] if key else ""
            content = meta.get("content", "")
            content_str = (
                content if isinstance(content, str) else content[0] if content else ""
            )
            if key_str:
                metadata[key_str] = content_str

    return ParsedDocument(title=title, text=text, links=links, metadata=metadata)
