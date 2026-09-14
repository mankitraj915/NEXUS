import heapq
import math
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.documents.models import Document, DocumentVersion
from app.domains.search.analyzer import analyze_text
from app.domains.search.models import DocumentStats, Posting, ProjectStats, Term
from app.domains.search.schemas import SearchResultItem

K1 = 1.5
B = 0.75


async def search_project(
    db: AsyncSession, project_id: uuid.UUID, query: str, top_k: int = 10
) -> list[SearchResultItem]:
    query_terms = analyze_text(query)
    if not query_terms:
        return []

    stmt_stats = select(ProjectStats).where(ProjectStats.project_id == project_id)
    proj_stats = (await db.execute(stmt_stats)).scalars().first()

    if not proj_stats or proj_stats.total_documents == 0:
        return []

    N = proj_stats.total_documents
    avgdl = proj_stats.total_tokens / N

    stmt_terms = select(Term).where(
        Term.project_id == project_id, Term.term.in_(query_terms)
    )
    terms = (await db.execute(stmt_terms)).scalars().all()

    if not terms:
        return []

    term_map = {t.term: t for t in terms}

    idfs = {}
    term_ids = []
    for term in query_terms:
        if term in term_map:
            t = term_map[term]
            n_q = t.document_frequency
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            idfs[t.id] = idf
            term_ids.append(t.id)

    if not term_ids:
        return []

    stmt_postings = (
        select(
            Posting.document_version_id,
            Posting.term_id,
            Posting.term_frequency,
            Posting.positions,
            DocumentStats.token_count,
        )
        .join(
            DocumentStats,
            Posting.document_version_id == DocumentStats.document_version_id,
        )
        .where(Posting.term_id.in_(term_ids), DocumentStats.project_id == project_id)
    )

    candidate_postings = (await db.execute(stmt_postings)).all()

    docs = {}
    for row in candidate_postings:
        doc_vid = row.document_version_id
        if doc_vid not in docs:
            docs[doc_vid] = {
                "token_count": row.token_count,
                "term_freqs": {},
                "min_pos": float("inf"),
            }
        docs[doc_vid]["term_freqs"][row.term_id] = row.term_frequency
        if row.positions:
            docs[doc_vid]["min_pos"] = min(docs[doc_vid]["min_pos"], min(row.positions))

    scores = []
    for doc_vid, data in docs.items():
        doc_len = data["token_count"]
        score = 0.0

        for term_id in term_ids:
            tf = data["term_freqs"].get(term_id, 0)
            if tf > 0:
                idf = idfs[term_id]
                numerator = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * (doc_len / avgdl))
                score += idf * (numerator / denominator)

        if score > 0:
            scores.append((score, doc_vid))

    top_results = heapq.nlargest(top_k, scores, key=lambda x: x[0])

    if not top_results:
        return []

    top_vids = [res[1] for res in top_results]
    score_map = {res[1]: res[0] for res in top_results}

    stmt_enrich = (
        select(DocumentVersion, Document)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(DocumentVersion.id.in_(top_vids))
    )

    enrich_results = (await db.execute(stmt_enrich)).all()

    final_results = []
    for ver, doc in enrich_results:
        min_pos = docs[ver.id]["min_pos"]
        if min_pos == float("inf"):
            min_pos = 0

        snippet = ""
        if ver.extracted_text:
            words = ver.extracted_text.split()
            start_idx = max(0, min_pos - 20)
            end_idx = min(len(words), min_pos + 40)
            snippet = " ".join(words[start_idx:end_idx]).replace("\n", " ") + "..."

        final_results.append(
            SearchResultItem(
                document_id=str(doc.id),
                url=doc.canonical_url,
                title=ver.raw_metadata.get("title", "Untitled")
                if ver.raw_metadata
                else "Untitled",
                score=round(score_map[ver.id], 4),
                snippet=snippet,
            )
        )

    final_results.sort(key=lambda x: x.score, reverse=True)
    return final_results
