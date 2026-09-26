from app.domains.search.retrieval.models import Candidate


def reciprocal_rank_fusion(
    bm25_candidates: list[Candidate], dense_candidates: list[Candidate], k: int = 60
) -> list[Candidate]:
    """
    Performs Reciprocal Rank Fusion on two lists of candidates.
    Returns a unified list of Candidates, ranked by rrf_score.
    """
    union_map: dict[str, Candidate] = {}

    for c in bm25_candidates:
        if c.document_id not in union_map:
            union_map[c.document_id] = Candidate(
                document_id=c.document_id, url=c.url, title=c.title, snippet=c.snippet
            )
        target = union_map[c.document_id]
        target.bm25_rank = c.bm25_rank
        target.bm25_score = c.bm25_score
        target.bm25_passage = c.bm25_passage
        if target.snippet is None:
            target.snippet = c.snippet

    for c in dense_candidates:
        if c.document_id not in union_map:
            union_map[c.document_id] = Candidate(
                document_id=c.document_id, url=c.url, title=c.title, snippet=c.snippet
            )
        target = union_map[c.document_id]
        target.dense_rank = c.dense_rank
        target.dense_score = c.dense_score
        target.dense_chunk_id = c.dense_chunk_id
        target.dense_passage = c.dense_passage
        # Dense chunk text provides more focused passage context than full document snippet
        target.snippet = c.snippet

    results = list(union_map.values())
    for c in results:
        rrf = 0.0
        if c.bm25_rank is not None:
            rrf += 1.0 / (k + c.bm25_rank)
        if c.dense_rank is not None:
            rrf += 1.0 / (k + c.dense_rank)
        c.rrf_score = rrf

    results.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)

    for i, c in enumerate(results):
        c.final_rank = i + 1

    return results


def multi_query_reciprocal_rank_fusion(
    variant_results: dict[str, dict[str, list[Candidate]]],
    k: int = 60,
    strategy: str = "strategy_a",
    weights: dict[str, float] | None = None,
) -> list[Candidate]:
    """
    variant_results is a dict: variant_id -> {"bm25": [Candidates], "dense": [Candidates]}

    Strategy A (Global Union RRF):
    All ranked lists (Q0-BM25, Q0-Dense, Q1-BM25, ...) are fused simultaneously.

    Strategy B (Two-Stage RRF):
    Fuse BM25 & Dense per variant first. Then fuse the resulting variant-level lists.
    """
    from app.domains.search.retrieval.models import VariantProvenance

    if weights is None:
        weights = {}

    if strategy == "strategy_b":
        # Strategy B: Two-Stage RRF
        variant_level_results = {}
        for v_id, res in variant_results.items():
            # First stage: standard RRF for this variant
            fused = reciprocal_rank_fusion(
                res.get("bm25", []), res.get("dense", []), k=k
            )
            # Re-map provenances to tag with this variant id
            for c in fused:
                if c.bm25_rank:
                    c.provenance.append(
                        VariantProvenance(
                            variant_id=v_id,
                            source="bm25",
                            rank=c.bm25_rank,
                            score=c.bm25_score or 0.0,
                            passage=c.bm25_passage,
                        )
                    )
                if c.dense_rank:
                    c.provenance.append(
                        VariantProvenance(
                            variant_id=v_id,
                            source="dense",
                            rank=c.dense_rank,
                            score=c.dense_score or 0.0,
                            passage=c.dense_passage,
                            chunk_id=c.dense_chunk_id,
                        )
                    )
            variant_level_results[v_id] = fused

        # Second stage: fuse the variant lists
        union_map: dict[str, Candidate] = {}
        for v_id, c_list in variant_level_results.items():
            for i, c in enumerate(c_list):
                rank = i + 1
                if c.document_id not in union_map:
                    union_map[c.document_id] = Candidate(
                        document_id=c.document_id,
                        url=c.url,
                        title=c.title,
                        snippet=c.snippet,
                    )
                target = union_map[c.document_id]
                # Accumulate provenances
                target.provenance.extend(c.provenance)
                # Ensure snippet is carried over
                if not target.snippet:
                    target.snippet = c.snippet

                # In Strategy B, we add 1 / (k + rank) for each variant ranking
                current_score = getattr(target, "rrf_score", 0.0) or 0.0
                variant_weight = weights.get(v_id, 1.0)
                target.rrf_score = current_score + (variant_weight / (k + rank))

        results = list(union_map.values())
        results.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)
        for i, c in enumerate(results):
            c.final_rank = i + 1
        return results

    # Default: Strategy A (Global Union RRF)
    global_union_map: dict[str, Candidate] = {}

    for v_id, res in variant_results.items():
        bm25_list = res.get("bm25", [])
        dense_list = res.get("dense", [])

        for c in bm25_list:
            if c.document_id not in global_union_map:
                global_union_map[c.document_id] = Candidate(
                    document_id=c.document_id,
                    url=c.url,
                    title=c.title,
                    snippet=c.snippet,
                )
            target = global_union_map[c.document_id]
            target.provenance.append(
                VariantProvenance(
                    variant_id=v_id,
                    source="bm25",
                    rank=c.bm25_rank or 0,
                    score=c.bm25_score or 0.0,
                    passage=c.bm25_passage,
                )
            )
            if not target.snippet:
                target.snippet = c.snippet

        for c in dense_list:
            if c.document_id not in global_union_map:
                global_union_map[c.document_id] = Candidate(
                    document_id=c.document_id,
                    url=c.url,
                    title=c.title,
                    snippet=c.snippet,
                )
            target = global_union_map[c.document_id]
            target.provenance.append(
                VariantProvenance(
                    variant_id=v_id,
                    source="dense",
                    rank=c.dense_rank or 0,
                    score=c.dense_score or 0.0,
                    passage=c.dense_passage,
                    chunk_id=c.dense_chunk_id,
                )
            )
            target.snippet = c.snippet  # Dense snippet is preferred

    # Calculate global RRF
    for c in global_union_map.values():
        rrf = 0.0
        for prov in c.provenance:
            variant_weight = weights.get(prov.variant_id, 1.0)
            rrf += variant_weight / (k + prov.rank)
        c.rrf_score = rrf

    results = list(global_union_map.values())
    results.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)
    for i, c in enumerate(results):
        c.final_rank = i + 1

    return results
