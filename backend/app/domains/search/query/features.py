import math
import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.search.analyzer import analyze_text
from app.domains.search.models import ProjectStats, Term
from app.domains.search.query.models import QueryFeatures

# Patterns for identifier detection
IDENTIFIER_PATTERNS = [
    re.compile(r"^[A-Z][a-zA-Z0-9]+[A-Z][a-zA-Z0-9]+"),  # CamelCase / PascalCase
    re.compile(r"^[a-z]+_[a-z0-9_]+$"),  # snake_case
    re.compile(
        r"^[a-zA-Z0-9]+\.[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*$"
    ),  # Dotted notation (e.g. asyncio.gather)
    re.compile(r"^/[a-zA-Z0-9_/-]+$"),  # File paths
    re.compile(r"^v?\d+\.\d+(\.\d+)?(-[a-zA-Z0-9]+)?$"),  # Semantic versions
    re.compile(r"^[A-Z0-9]{3,}$"),  # Acronyms
]

WH_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "who",
    "which",
    "where",
    "is",
    "are",
    "do",
    "does",
    "can",
    "could",
    "should",
    "would",
}
STOP_WORDS = {
    "a",
    "an",
    "the",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "as",
    "about",
}
CONJUNCTIONS = {"and", "or", "vs", "versus", "compare", "between"}


async def compute_term_rarity(
    db: AsyncSession, project_id: uuid.UUID, tokens: list[str]
) -> float:
    """
    Computes a corpus-derived rarity score using BM25 index statistics.
    Returns a score between [0, 1].
    1.0 = extremely rare (high IDF)
    0.0 = very common or not found in corpus
    """
    if not tokens:
        return 0.0

    stmt_stats = select(ProjectStats.total_documents).where(
        ProjectStats.project_id == project_id
    )
    total_docs = (await db.execute(stmt_stats)).scalar() or 0
    if total_docs == 0:
        return 0.0

    stmt_terms = select(Term.term, Term.document_frequency).where(
        Term.project_id == project_id, Term.term.in_(tokens)
    )
    terms = (await db.execute(stmt_terms)).all()

    if not terms:
        return 0.0

    # Calculate IDF for each term: log((N - n + 0.5) / (n + 0.5) + 1)
    # Using a simpler normalized IDF: log(N / df) / log(N)
    # This guarantees [0, 1] range.
    max_idf_possible = math.log(total_docs + 1)
    if max_idf_possible <= 0:
        return 0.0

    idfs = []
    for term, df in terms:
        # Avoid zero or negative
        df = max(1, df)
        # Standard normalized rarity
        rarity = math.log(total_docs / df) / max_idf_possible
        idfs.append(max(0.0, min(1.0, rarity)))

    # We take the 90th percentile rarity (or max) to represent the most rare terms driving the query
    idfs.sort()
    return idfs[-1] if idfs else 0.0


def extract_features(raw_query: str, db_rarity_score: float = 0.0) -> QueryFeatures:
    # Use raw tokens (with stopwords intact) for feature extraction
    raw_tokens_lower = analyze_text(raw_query, remove_stopwords=False)

    # We still want token count based on the original words
    token_count = len(raw_tokens_lower)

    if token_count == 0:
        return QueryFeatures()

    unique_tokens = set(raw_tokens_lower)
    unique_token_ratio = len(unique_tokens) / token_count

    # Quoted term density
    # Count words inside quotes
    quoted_matches = re.findall(r'"([^"]+)"', raw_query)
    quoted_text = " ".join(quoted_matches)
    quoted_tokens = analyze_text(quoted_text)
    quoted_term_density = min(1.0, len(quoted_tokens) / token_count)

    # Identifier density (use raw space-split to capture CamelCase etc before lowering/stemming)
    raw_tokens = raw_query.split()
    identifier_count = 0
    for rt in raw_tokens:
        clean_rt = rt.strip(".,!?;:()[]{}\"'")
        if any(pat.match(clean_rt) for pat in IDENTIFIER_PATTERNS):
            identifier_count += 1

    identifier_density = (
        min(1.0, identifier_count / len(raw_tokens)) if raw_tokens else 0.0
    )

    # Lexical specificity
    # Driven by exact matches, identifiers, and rare terms
    lexical_specificity_score = min(
        1.0,
        (
            (quoted_term_density * 0.4)
            + (identifier_density * 0.4)
            + (db_rarity_score * 0.2)
        ),
    )

    # Natural language score
    wh_count = sum(1 for t in raw_tokens_lower if t in WH_WORDS)
    stop_count = sum(1 for t in raw_tokens_lower if t in STOP_WORDS)
    has_question_mark = 1.0 if "?" in raw_query else 0.0

    # High if it has question words, stopwords, and punctuation
    nl_components = []
    nl_components.append(min(1.0, wh_count / 2.0))  # 1-2 WH words is highly indicative
    nl_components.append(
        min(1.0, stop_count / (token_count * 0.5 + 0.1))
    )  # If ~50% stopwords, very NL
    nl_components.append(has_question_mark * 0.5)

    natural_language_score = min(1.0, sum(nl_components) / 2.5)  # Normalize

    # Complexity
    conjunction_count = sum(1 for t in raw_tokens_lower if t in CONJUNCTIONS)
    length_penalty = min(1.0, token_count / 15.0)  # Longer than 15 tokens is complex

    complexity_score = min(1.0, ((conjunction_count * 0.3) + (length_penalty * 0.7)))

    return QueryFeatures(
        token_count=token_count,
        unique_token_ratio=unique_token_ratio,
        identifier_density=identifier_density,
        quoted_term_density=quoted_term_density,
        term_rarity_score=db_rarity_score,
        lexical_specificity_score=lexical_specificity_score,
        natural_language_score=natural_language_score,
        complexity_score=complexity_score,
    )
