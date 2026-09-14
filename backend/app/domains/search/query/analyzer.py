import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.search.analyzer import analyze_text
from app.domains.search.query.classifier import classify_query
from app.domains.search.query.features import compute_term_rarity, extract_features
from app.domains.search.query.models import QueryAnalysis
from app.domains.search.query.planner import generate_plan

logger = logging.getLogger(__name__)


async def analyze_query(
    db: AsyncSession, project_id: uuid.UUID, raw_query: str
) -> QueryAnalysis:
    """
    Deterministically analyzes a query, extracts features, classifies it, and generates a retrieval plan.
    """
    normalized_query = " ".join(analyze_text(raw_query))

    # 1. Fetch corpus-derived term rarity statistics
    tokens = analyze_text(raw_query)
    try:
        db_rarity_score = await compute_term_rarity(db, project_id, tokens)
    except Exception as e:
        logger.error(f"Failed to compute term rarity: {e}")
        db_rarity_score = 0.0

    # 2. Extract deterministic features
    features = extract_features(raw_query, db_rarity_score)

    # 3. Classify query (multi-label)
    classes = classify_query(features)

    # 4. Generate plan
    plan = generate_plan(features, classes)

    return QueryAnalysis(
        raw_query=raw_query,
        normalized_query=normalized_query,
        features=features,
        classes=classes,
        plan=plan,
    )
