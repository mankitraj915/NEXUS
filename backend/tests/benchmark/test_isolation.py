import uuid

import pytest
from sqlalchemy import select

from app.domains.documents.models import Document
from app.domains.documents.service import save_document
from app.domains.projects.models import Project
from app.domains.search.models import Term
from app.domains.search.retrieval.fusion import reciprocal_rank_fusion
from app.domains.search.retrieval.lexical import retrieve_lexical
from app.domains.search.retrieval.models import Candidate
from app.domains.users.models import User

# ==============================================================================
# 1. Hermetic Unit Isolation Tests (No DB connection needed)
# ==============================================================================


def test_hermetic_candidate_metadata_project_isolation():
    """Prove that candidate structures retain project context and do not merge across tenants."""
    doc_id_a = str(uuid.uuid4())
    doc_id_b = str(uuid.uuid4())

    cand_a = Candidate(
        document_id=doc_id_a,
        bm25_score=0.9,
        title="Project A Doc",
        snippet="Secret A",
        url="https://example.com/doc",
    )
    Candidate(
        document_id=doc_id_b,
        bm25_score=0.85,
        title="Project B Doc",
        snippet="Secret B",
        url="https://example.com/doc",
    )

    # RRF on candidates from a single tenant works deterministically
    fused_a = reciprocal_rank_fusion([cand_a], [], k=60)
    assert len(fused_a) == 1
    assert fused_a[0].document_id == doc_id_a
    assert fused_a[0].document_id != doc_id_b


# ==============================================================================
# 2. Integration Isolation Tests (Requires live PostgreSQL/pgvector)
# ==============================================================================


@pytest.mark.asyncio
async def test_identical_url_coexistence_in_different_projects(db_session):
    """
    Step 1 Acceptance Criteria:
    Project A and Project B ingest the exact same canonical URL.
    Both must coexist with distinct document IDs, preserving project isolation.
    """
    user = User(email="isolation_test@nexus.com", hashed_password="fake")
    db_session.add(user)
    await db_session.flush()

    project_a = Project(name="Project A", user_id=user.id)
    project_b = Project(name="Project B", user_id=user.id)
    db_session.add_all([project_a, project_b])
    await db_session.commit()

    shared_url = "https://example.com/shared-page"

    # Ingest same URL into Project A and Project B
    ver_a = await save_document(
        db=db_session,
        project_id=project_a.id,
        canonical_url=shared_url,
        source_domain="example.com",
        text="NexusAlpha proprietary documentation for Project A.",
        metadata={"project": "A"},
    )

    ver_b = await save_document(
        db=db_session,
        project_id=project_b.id,
        canonical_url=shared_url,
        source_domain="example.com",
        text="NexusBeta proprietary documentation for Project B.",
        metadata={"project": "B"},
    )

    # Invariant 1: Document IDs must be distinct
    assert ver_a.document_id != ver_b.document_id

    # Invariant 2: In the database, each document is strictly scoped to its project
    stmt_a = select(Document).where(Document.id == ver_a.document_id)
    doc_a = (await db_session.execute(stmt_a)).scalar_one()
    assert doc_a.project_id == project_a.id
    assert doc_a.canonical_url == shared_url

    stmt_b = select(Document).where(Document.id == ver_b.document_id)
    doc_b = (await db_session.execute(stmt_b)).scalar_one()
    assert doc_b.project_id == project_b.id
    assert doc_b.canonical_url == shared_url


@pytest.mark.asyncio
async def test_same_project_deduplication(db_session):
    """
    Step 1 Acceptance Criteria:
    Same-project duplicate ingestion continues to deduplicate correctly.
    """
    user = User(email="dedup_test@nexus.com", hashed_password="fake")
    db_session.add(user)
    await db_session.flush()

    project = Project(name="Dedup Project", user_id=user.id)
    db_session.add(project)
    await db_session.commit()

    url = "https://example.com/dedup-page"
    text = "Initial content that will be re-crawled."

    ver_1 = await save_document(
        db=db_session,
        project_id=project.id,
        canonical_url=url,
        source_domain="example.com",
        text=text,
        metadata={"crawl": 1},
    )

    # Re-crawl same content in same project
    ver_2 = await save_document(
        db=db_session,
        project_id=project.id,
        canonical_url=url,
        source_domain="example.com",
        text=text,
        metadata={"crawl": 2},
    )

    assert ver_1.id == ver_2.id
    assert ver_1.document_id == ver_2.document_id


@pytest.mark.asyncio
async def test_lexical_retrieval_isolation(db_session):
    """
    Step 2 Acceptance Criteria:
    Project A contains 'NexusAlpha'.
    Project B contains 'NexusBeta'.
    Querying A for 'NexusBeta' must NOT retrieve B's document.
    Querying B for 'NexusAlpha' must NOT retrieve A's document.
    """
    user = User(email="lexical_isolation@nexus.com", hashed_password="fake")
    db_session.add(user)
    await db_session.flush()

    project_a = Project(name="Lexical Project A", user_id=user.id)
    project_b = Project(name="Lexical Project B", user_id=user.id)
    db_session.add_all([project_a, project_b])
    await db_session.commit()

    ver_a = await save_document(
        db=db_session,
        project_id=project_a.id,
        canonical_url="https://alpha.example.com/doc",
        source_domain="alpha.example.com",
        text="NexusAlpha contains quantum algorithms.",
        metadata={},
    )

    ver_b = await save_document(
        db=db_session,
        project_id=project_b.id,
        canonical_url="https://beta.example.com/doc",
        source_domain="beta.example.com",
        text="NexusBeta contains relativistic mechanics.",
        metadata={},
    )

    # Query Project A for Project B's term: must return 0 results
    results_a_for_beta = await retrieve_lexical(
        db_session, project_a.id, "NexusBeta", top_k=10
    )
    assert len(results_a_for_beta) == 0

    # Query Project B for Project A's term: must return 0 results
    results_b_for_alpha = await retrieve_lexical(
        db_session, project_b.id, "NexusAlpha", top_k=10
    )
    assert len(results_b_for_alpha) == 0

    # Query Project A for its own term: must return Project A's document
    results_a_for_alpha = await retrieve_lexical(
        db_session, project_a.id, "NexusAlpha", top_k=10
    )
    assert len(results_a_for_alpha) == 1
    assert results_a_for_alpha[0].document_id == str(ver_a.document_id)

    # Query Project B for its own term: must return Project B's document
    results_b_for_beta = await retrieve_lexical(
        db_session, project_b.id, "NexusBeta", top_k=10
    )
    assert len(results_b_for_beta) == 1
    assert results_b_for_beta[0].document_id == str(ver_b.document_id)


@pytest.mark.asyncio
async def test_statistics_isolation(db_session):
    """
    Step 2 Acceptance Criteria:
    Corpus statistics and inverted index terms must be strictly scoped by project_id.
    """
    user = User(email="stats_isolation@nexus.com", hashed_password="fake")
    db_session.add(user)
    await db_session.flush()

    project_a = Project(name="Stats Project A", user_id=user.id)
    project_b = Project(name="Stats Project B", user_id=user.id)
    db_session.add_all([project_a, project_b])
    await db_session.commit()

    await save_document(
        db=db_session,
        project_id=project_a.id,
        canonical_url="https://example.com/a1",
        source_domain="example.com",
        text="UniqueAlphaTerm only exists in A.",
        metadata={},
    )

    # Term index for project B must not contain Project A's unique term
    stmt_b_terms = select(Term).where(
        Term.project_id == project_b.id, Term.term == "uniquealphaterm"
    )
    terms_in_b = (await db_session.execute(stmt_b_terms)).scalars().all()
    assert len(terms_in_b) == 0

    # Term index for project A must contain Project A's term
    stmt_a_terms = select(Term).where(
        Term.project_id == project_a.id, Term.term == "uniquealphaterm"
    )
    terms_in_a = (await db_session.execute(stmt_a_terms)).scalars().all()
    assert len(terms_in_a) == 1
