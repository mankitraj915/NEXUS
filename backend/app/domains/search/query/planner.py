from app.domains.search.query.models import QueryClasses, QueryFeatures, RetrievalPlan


def generate_plan(features: QueryFeatures, classes: QueryClasses) -> RetrievalPlan:
    # Default Phase 5 Production Baseline
    base_bm25 = 100
    base_dense = 100
    base_rrf = 10

    plan = RetrievalPlan(
        bm25_depth=base_bm25,
        dense_depth=base_dense,
        rrf_k=base_rrf,
        reason="Default hybrid configuration",
        policy_version="1.0.0",
    )

    # Policy rules evaluated in order of precedence

    if classes.EXACT_TERM or classes.IDENTIFIER_HEAVY:
        plan.bm25_depth = 100
        plan.dense_depth = 25
        plan.reason = "High lexical specificity and identifier density"

    elif classes.NATURAL_LANGUAGE_QUESTION or classes.CONCEPTUAL:
        # If it's a mixed query (technical terms but asked as a question),
        # we don't want to completely drop BM25.
        if classes.TECHNICAL:
            plan.bm25_depth = 100
            plan.dense_depth = 100
            plan.reason = "Mixed semantic and technical signals"
        else:
            plan.bm25_depth = 25
            plan.dense_depth = 100
            plan.reason = "Conceptual/NL query with low lexical specificity"

    elif classes.AMBIGUOUS:
        # For ambiguous queries, keep both high to maximize chance of hitting relevant chunk
        plan.bm25_depth = 100
        plan.dense_depth = 100
        plan.reason = "Ambiguous short query"

    # Validation against pathological depths
    MAX_DEPTH = 500
    plan.bm25_depth = max(1, min(MAX_DEPTH, plan.bm25_depth))
    plan.dense_depth = max(1, min(MAX_DEPTH, plan.dense_depth))
    plan.rrf_k = max(1, min(200, plan.rrf_k))

    return plan
