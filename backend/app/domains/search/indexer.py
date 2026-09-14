from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from app.domains.documents.models import DocumentVersion, Document
from app.domains.search.models import ProjectStats, DocumentStats, Term, Posting
from app.domains.search.analyzer import analyze_text
from typing import List, Dict
import uuid
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

async def index_document_version(db: AsyncSession, version_id: uuid.UUID):
    stmt = select(DocumentVersion, Document.project_id).join(Document).where(DocumentVersion.id == version_id)
    result = await db.execute(stmt)
    row = result.first()
    if not row:
        logger.warning(f"DocumentVersion {version_id} not found.")
        return
    
    version, project_id = row[0], row[1]
    
    stmt = select(DocumentStats).where(DocumentStats.document_version_id == version_id)
    doc_stats = (await db.execute(stmt)).scalars().first()
    if doc_stats and doc_stats.status == "INDEXED":
        return
        
    if not doc_stats:
        doc_stats = DocumentStats(document_version_id=version_id, project_id=project_id, status="PENDING")
        db.add(doc_stats)
        await db.commit()
        await db.refresh(doc_stats)
        
    try:
        tokens = analyze_text(version.extracted_text)
        token_count = len(tokens)
        
        tf = defaultdict(int)
        positions = defaultdict(list)
        for pos, token in enumerate(tokens):
            tf[token] += 1
            positions[token].append(pos)
            
        # 3. Update Project Stats (Upsert)
        proj_stmt = insert(ProjectStats).values(
            project_id=project_id, total_documents=1, total_tokens=token_count
        ).on_conflict_do_update(
            index_elements=["project_id"],
            set_=dict(
                total_documents=ProjectStats.total_documents + 1,
                total_tokens=ProjectStats.total_tokens + token_count
            )
        )
        await db.execute(proj_stmt)
            
        # 4. Upsert Terms and Create Postings
        if tf:
            term_values = [
                dict(project_id=project_id, term=term_str, document_frequency=1)
                for term_str in tf.keys()
            ]
            term_stmt = insert(Term).values(term_values).on_conflict_do_update(
                index_elements=["project_id", "term"],
                set_=dict(document_frequency=Term.document_frequency + 1)
            ).returning(Term.id, Term.term)
            
            result = await db.execute(term_stmt)
            term_rows = result.all()
            
            postings_data = [
                dict(
                    term_id=row.id,
                    document_version_id=version_id,
                    term_frequency=tf[row.term],
                    positions=positions[row.term]
                )
                for row in term_rows
            ]
            
            if postings_data:
                await db.execute(insert(Posting).values(postings_data))
            
        # 5. Update DocumentStats
        doc_stats.token_count = token_count
        doc_stats.status = "INDEXED"
        
        await db.commit()
        
    except Exception as e:
        logger.error(f"Failed to index {version_id}: {e}")
        doc_stats.status = "FAILED"
        await db.commit()
