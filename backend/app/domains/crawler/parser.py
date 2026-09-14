from bs4 import BeautifulSoup
from typing import Dict, List
from urllib.parse import urljoin

class ParsedDocument:
    def __init__(self, title: str, text: str, links: List[str], metadata: Dict[str, str]):
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
    
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(("http", "https", "/")):
            links.append(urljoin(base_url, href))
            
    metadata = {}
    for meta in soup.find_all("meta"):
        if meta.get("name"):
            metadata[meta.get("name")] = meta.get("content", "")
        elif meta.get("property"):
            metadata[meta.get("property")] = meta.get("content", "")
            
    return ParsedDocument(title=title, text=text, links=links, metadata=metadata)
