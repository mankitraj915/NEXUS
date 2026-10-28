import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.documents.models import DocumentVersion
from app.domains.documents.service import save_document
from app.domains.embeddings.models import Chunk
from app.domains.projects.models import Project
from app.domains.search.analyzer import analyze_text
from app.domains.search.models import DocumentStats, Posting, ProjectStats, Term
from app.domains.search.retrieval.lexical import retrieve_lexical
from app.domains.users.models import User

URL = "https://docs.example.com/page"
OLD_TEXT = "Glacier hydrology describes meltwater channels beneath ice sheets."
NEW_TEXT = "Volcanic tephra layers record eruption chronology in lake sediments."


async def _project(db: AsyncSession, email: str) -> Project:
    user = User(email=email, hashed_password="pw")
    db.add(user)
    await db.flush()
    project = Project(name="Consistency", user_id=user.id)
    db.add(project)
    await db.commit()
    return project


async def _count(db: AsyncSession, stmt) -> int:
    return (await db.execute(stmt)).scalar_one()


@pytest.mark.asyncio
async def test_resaving_identical_content_is_idempotent(db_session):
    project = await _project(db_session, "idempotent@nexus.com")

    v1 = await save_document(db_session, project.id, URL, "example.com", OLD_TEXT, {})
    v2 = await save_document(db_session, project.id, URL, "example.com", OLD_TEXT, {})

    assert v1.id == v2.id
    stats = await db_session.get(ProjectStats, project.id)
    assert stats is not None
    assert stats.total_documents == 1
    assert stats.total_tokens == len(analyze_text(OLD_TEXT))


@pytest.mark.asyncio
async def test_content_change_replaces_previous_version_in_index(db_session):
    project = await _project(db_session, "reindex@nexus.com")

    old = await save_document(db_session, project.id, URL, "example.com", OLD_TEXT, {})
    new = await save_document(db_session, project.id, URL, "example.com", NEW_TEXT, {})
    assert old.id != new.id
    assert old.document_id == new.document_id

    # Corpus statistics count the document once, with the new version's length
    stats = await db_session.get(ProjectStats, project.id)
    await db_session.refresh(stats)
    assert stats.total_documents == 1
    assert stats.total_tokens == len(analyze_text(NEW_TEXT))

    # Only the new version is indexed (postings, stats, chunks); history is kept
    assert (
        await _count(
            db_session,
            select(func.count())
            .select_from(DocumentVersion)
            .where(DocumentVersion.document_id == new.document_id),
        )
        == 2
    )
    for model, column in (
        (Posting, Posting.document_version_id),
        (DocumentStats, DocumentStats.document_version_id),
        (Chunk, Chunk.document_version_id),
    ):
        assert (
            await _count(
                db_session,
                select(func.count()).select_from(model).where(column == old.id),
            )
            == 0
        )

    # Terms that only appeared in the old text no longer count the document
    old_only = set(analyze_text(OLD_TEXT)) - set(analyze_text(NEW_TEXT))
    dfs = (
        await db_session.execute(
            select(Term.term, Term.document_frequency).where(
                Term.project_id == project.id, Term.term.in_(old_only)
            )
        )
    ).all()
    await db_session.commit()
    assert dfs and all(df == 0 for _, df in dfs)

    # Old content is not retrievable; new content is retrievable exactly once
    assert await retrieve_lexical(db_session, project.id, "glacier meltwater") == []
    hits = await retrieve_lexical(db_session, project.id, "volcanic tephra eruption")
    assert [h.document_id for h in hits] == [str(new.document_id)]


@pytest.mark.asyncio
async def test_lexical_edge_cases_return_empty(db_session):
    project = await _project(db_session, "edges@nexus.com")

    # Empty project: no statistics yet
    assert await retrieve_lexical(db_session, project.id, "anything") == []

    await save_document(db_session, project.id, URL, "example.com", OLD_TEXT, {})

    # Stopword-only and punctuation-only queries analyze to no terms
    assert await retrieve_lexical(db_session, project.id, "the and of") == []
    assert await retrieve_lexical(db_session, project.id, "?!...") == []
    # Terms absent from the vocabulary
    assert await retrieve_lexical(db_session, project.id, "zeolite") == []


@pytest.mark.asyncio
async def test_lexical_ranking_is_sorted_and_deterministic(db_session):
    project = await _project(db_session, "ranking@nexus.com")
    texts = [
        "Retrieval evaluation with graded relevance judgments.",
        "Retrieval retrieval retrieval systems and retrieval evaluation.",
        "Unrelated text about bread fermentation and sourdough.",
        "Retrieval systems.",
    ]
    for i, text in enumerate(texts):
        await save_document(
            db_session, project.id, f"{URL}/{i}", "example.com", text, {}
        )

    first = await retrieve_lexical(db_session, project.id, "retrieval evaluation")
    second = await retrieve_lexical(db_session, project.id, "retrieval evaluation")

    scores = [c.bm25_score for c in first if c.bm25_score is not None]
    assert len(scores) == len(first)
    assert scores == sorted(scores, reverse=True)
    assert [c.bm25_rank for c in first] == list(range(1, len(first) + 1))
    assert len({c.document_id for c in first}) == len(first) == 3
    assert [c.document_id for c in first] == [c.document_id for c in second]
