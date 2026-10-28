import asyncio
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.search.query.analyzer import analyze_query
from app.domains.search.retrieval.dense import retrieve_dense
from app.domains.search.retrieval.fusion import reciprocal_rank_fusion
from app.domains.search.retrieval.lexical import retrieve_lexical
from app.domains.search.retrieval.models import Candidate
from app.domains.search.retrieval.reranker import LocalCrossEncoderReranker

logger = logging.getLogger(__name__)

# Single instance of reranker to share loaded model
_reranker = LocalCrossEncoderReranker()


async def execute_search(
    db: AsyncSession,
    project_id: uuid.UUID,
    query: str,
    mode: str = "hybrid_fixed",
    top_k: int = 10,
    retrieval_k: int = 100,
    retrieval_k_bm25: int | None = None,
    retrieval_k_dense: int | None = None,
    rerank_k: int = 25,
    rrf_k: int = 10,
    return_analysis: bool = False,
    multi_query_strategy: str = "strategy_a",
    llm_max_variants: int = 3,
    llm_variant_weight: float = 1.0,
    llm_drift_threshold: float = 0.70,
):
    """
    Executes search and optionally returns analysis data if requested.
    Returns (List[Candidate], Optional[QueryAnalysis])
    """

    from app.domains.search.schemas import SearchMode

    mode = SearchMode.normalize_mode(mode)

    analysis = None

    if mode == "adaptive":
        try:
            analysis = await analyze_query(db, project_id, query)
            if analysis and analysis.plan:
                plan = analysis.plan
                retrieval_k_bm25 = plan.bm25_depth
                retrieval_k_dense = plan.dense_depth
                rrf_k = plan.rrf_k
                logger.info(
                    f"Adaptive Plan for '{query}': BM25={retrieval_k_bm25}, Dense={retrieval_k_dense} | Class: {analysis.classes}"
                )
        except Exception as e:
            logger.error(f"Adaptive planning failed, falling back to hybrid_fixed: {e}")
            mode = "hybrid_fixed"

    elif mode == "learned":
        try:
            from app.domains.search.query.features import extract_features
            from app.domains.search.retrieval.learned_router import get_learned_router

            features = extract_features(query, db_rarity_score=0.1).model_dump()
            router = get_learned_router()
            plan_dict = router.predict(features)
            retrieval_k_bm25 = plan_dict["bm25_depth"]
            retrieval_k_dense = plan_dict["dense_depth"]
            rrf_k = plan_dict["rrf_k"]
            logger.info(
                f"Learned Plan for '{query}': BM25={retrieval_k_bm25}, Dense={retrieval_k_dense}"
            )
        except Exception as e:
            logger.error(f"Learned planning failed, falling back to hybrid_fixed: {e}")
            mode = "hybrid_fixed"

    if mode not in ("adaptive", "learned"):
        retrieval_k_bm25 = (
            retrieval_k_bm25 if retrieval_k_bm25 is not None else retrieval_k
        )
        retrieval_k_dense = (
            retrieval_k_dense if retrieval_k_dense is not None else retrieval_k
        )

    bm25_candidates = []
    dense_candidates = []

    if mode == "multi_query_rule":
        from app.domains.search.query.transformer import RuleBasedQueryTransformer
        from app.domains.search.retrieval.fusion import (
            multi_query_reciprocal_rank_fusion,
        )

        transformer = RuleBasedQueryTransformer()
        variants = await transformer.transform(query)

        tasks = []
        variant_results: dict[str, dict[str, list[Candidate]]] = {}
        for v in variants:
            variant_results[v.variant_id] = {"bm25": [], "dense": []}
            t_bm25 = asyncio.create_task(
                retrieve_lexical(
                    db, project_id, v.query, top_k=retrieval_k_bm25 or top_k
                )
            )
            t_dense = asyncio.create_task(
                retrieve_dense(
                    db, project_id, v.query, top_k=retrieval_k_dense or top_k
                )
            )
            tasks.append((v.variant_id, t_bm25, t_dense))

        for v_id, t_bm25, t_dense in tasks:
            try:
                variant_results[v_id]["bm25"] = await t_bm25
            except Exception as e:
                logger.error(f"BM25 failed for {v_id}: {e}")
            try:
                variant_results[v_id]["dense"] = await t_dense
            except Exception as e:
                logger.error(f"Dense failed for {v_id}: {e}")

        fused_candidates = multi_query_reciprocal_rank_fusion(
            variant_results, k=rrf_k, strategy=multi_query_strategy
        )
        res = fused_candidates[:top_k]

        # Add analysis object to track how many variants we generated
        from app.domains.search.query.models import (
            QueryAnalysis,
            QueryClasses,
            QueryFeatures,
        )

        analysis = QueryAnalysis(
            raw_query=query,
            normalized_query=query,
            features=QueryFeatures(token_count=len(variants)),
            classes=QueryClasses(),
        )
        return (res, analysis) if return_analysis else res

    if mode == "multi_query_llm":
        from app.domains.search.query.cache import VariantCache
        from app.domains.search.query.drift import SemanticDriftFilter
        from app.domains.search.query.llm_provider import OpenAICompatibleProvider
        from app.domains.search.query.transformer import LLMQueryTransformer
        from app.domains.search.retrieval.fusion import (
            multi_query_reciprocal_rank_fusion,
        )

        provider = OpenAICompatibleProvider(temperature=0.0)
        drift_filter = SemanticDriftFilter(threshold=llm_drift_threshold)
        cache = VariantCache()
        llm_transformer = LLMQueryTransformer(
            provider, drift_filter, cache, max_variants=llm_max_variants
        )

        variants = await llm_transformer.transform(query)

        # Parallel Retrieval
        tasks = []
        llm_variant_results: dict[str, dict[str, list[Candidate]]] = {}
        weights = {}
        for v in variants:
            weights[v.variant_id] = 1.0 if v.is_original else llm_variant_weight
            llm_variant_results[v.variant_id] = {"bm25": [], "dense": []}
            t_bm25 = asyncio.create_task(
                retrieve_lexical(
                    db, project_id, v.query, top_k=retrieval_k_bm25 or top_k
                )
            )
            t_dense = asyncio.create_task(
                retrieve_dense(
                    db, project_id, v.query, top_k=retrieval_k_dense or top_k
                )
            )
            tasks.append((v.variant_id, t_bm25, t_dense))

        for v_id, t_bm25, t_dense in tasks:
            try:
                llm_variant_results[v_id]["bm25"] = await t_bm25
            except Exception as e:
                logger.error(f"BM25 failed for {v_id}: {e}")
            try:
                llm_variant_results[v_id]["dense"] = await t_dense
            except Exception as e:
                logger.error(f"Dense failed for {v_id}: {e}")

        fused_candidates = multi_query_reciprocal_rank_fusion(
            llm_variant_results, k=rrf_k, strategy=multi_query_strategy, weights=weights
        )
        res = fused_candidates[:top_k]

        from app.domains.search.query.models import (
            QueryAnalysis,
            QueryClasses,
            QueryFeatures,
        )

        analysis = QueryAnalysis(
            raw_query=query,
            normalized_query=query,
            features=QueryFeatures(token_count=len(variants)),
            classes=QueryClasses(),
        )
        return (res, analysis) if return_analysis else res

    bm25_task = None
    dense_task = None

    if mode in [
        "lexical",
        "hybrid_fixed",
        "hybrid_no_rerank",
        "hybrid_rerank",
        "adaptive",
        "learned",
    ]:
        k_bm25 = (
            retrieval_k_bm25
            if (
                mode
                in (
                    "hybrid_fixed",
                    "adaptive",
                    "learned",
                    "hybrid_no_rerank",
                    "hybrid_rerank",
                )
                and retrieval_k_bm25 is not None
            )
            else top_k
        )
        bm25_task = asyncio.create_task(
            retrieve_lexical(db, project_id, query, top_k=k_bm25)
        )

    if mode in [
        "dense",
        "hybrid_fixed",
        "hybrid_no_rerank",
        "hybrid_rerank",
        "adaptive",
        "learned",
    ]:
        k_dense = (
            retrieval_k_dense
            if (
                mode
                in (
                    "hybrid_fixed",
                    "adaptive",
                    "learned",
                    "hybrid_no_rerank",
                    "hybrid_rerank",
                )
                and retrieval_k_dense is not None
            )
            else top_k
        )
        dense_task = asyncio.create_task(
            retrieve_dense(db, project_id, query, top_k=k_dense)
        )

    if bm25_task:
        try:
            bm25_candidates = await bm25_task
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            if mode == "lexical":
                return ([], analysis) if return_analysis else []

    if dense_task:
        try:
            dense_candidates = await dense_task
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            if mode == "dense":
                return ([], analysis) if return_analysis else []

    if mode == "lexical":
        return (bm25_candidates, analysis) if return_analysis else bm25_candidates
    elif mode == "dense":
        return (dense_candidates, analysis) if return_analysis else dense_candidates

    fused_candidates = []
    try:
        fused_candidates = reciprocal_rank_fusion(
            bm25_candidates, dense_candidates, k=rrf_k
        )
    except Exception as e:
        logger.error(f"RRF fusion failed: {e}")
        res = (bm25_candidates or dense_candidates)[:top_k]
        return (res, analysis) if return_analysis else res

    if mode in ["hybrid_no_rerank", "hybrid_fixed", "adaptive", "learned"]:
        res = fused_candidates[:top_k]
        return (res, analysis) if return_analysis else res

    if mode == "hybrid_rerank":
        try:
            candidates_to_rerank = fused_candidates[:rerank_k]
            reranked = _reranker.rerank(query, candidates_to_rerank, top_k=top_k)
            return (reranked, analysis) if return_analysis else reranked
        except Exception as e:
            logger.error(f"Reranker failed: {e}")
            res = fused_candidates[:top_k]
            return (res, analysis) if return_analysis else res

    return ([], analysis) if return_analysis else []
