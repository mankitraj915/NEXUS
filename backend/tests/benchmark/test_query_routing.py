from app.domains.search.query.classifier import classify_query
from app.domains.search.query.features import extract_features
from app.domains.search.query.planner import generate_plan


def test_extract_features_identifiers():
    query = "SQLAlchemy and MissingGreenlet in asyncio.gather"
    features = extract_features(query, 0.5)

    assert features.token_count > 0
    # Expected identifiers: SQLAlchemy, MissingGreenlet, asyncio.gather
    assert features.identifier_density > 0.3


def test_extract_features_quoted():
    query = 'What is "optimistic concurrency control"'
    features = extract_features(query, 0.2)
    assert features.quoted_term_density > 0.0


def test_extract_features_nl():
    query = "How does optimistic concurrency control work?"
    features = extract_features(query, 0.1)
    # how, does, work, ? should trigger a high NL score
    assert features.natural_language_score >= 0.4


def test_classification():
    query = "SQLAlchemy MissingGreenlet HTTP 429"
    features = extract_features(query, 0.8)
    classes = classify_query(features)

    assert classes.IDENTIFIER_HEAVY
    assert classes.TECHNICAL

    query2 = "How do I configure logging?"
    features2 = extract_features(query2, 0.1)
    classes2 = classify_query(features2)

    assert classes2.NATURAL_LANGUAGE_QUESTION


def test_planner_routing():
    # Lexical Heavy
    query = "SQLAlchemy MissingGreenlet HTTP 429"
    features = extract_features(query, 0.9)
    classes = classify_query(features)
    plan = generate_plan(features, classes)

    assert plan.bm25_depth == 100
    assert plan.dense_depth == 25
    assert plan.rrf_k == 10

    # NL Heavy
    query2 = "What is the meaning of life?"
    features2 = extract_features(query2, 0.1)
    classes2 = classify_query(features2)
    plan2 = generate_plan(features2, classes2)

    assert plan2.bm25_depth == 25
    assert plan2.dense_depth == 100
