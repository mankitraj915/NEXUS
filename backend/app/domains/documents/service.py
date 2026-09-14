from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.domains.documents.models import Document, DocumentVersion
from app.domains.crawler.models import CrawlTarget
from app.domains.search.indexer import index_document_version
import hashlib
import uuid
from typing import Dict, Optional

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

async def save_document(
    db: AsyncSession, 
    project_id: uuid.UUID, 
    canonical_url: str, 
    source_domain: str, 
    text: str, 
    metadata: Dict[str, str]
) -> DocumentVersion:
    stmt = select(Document).where(Document.canonical_url == canonical_url)
    result = await db.execute(stmt)
    doc = result.scalars().first()
    
    if not doc:
        doc = Document(canonical_url=canonical_url, source_domain=source_domain, project_id=project_id)
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        
    content_hash = compute_hash(text)
    
    stmt = select(DocumentVersion).where(DocumentVersion.document_id == doc.id).order_by(DocumentVersion.created_at.desc()).limit(1)
    result = await db.execute(stmt)
    latest_version = result.scalars().first()
    
    if latest_version and latest_version.content_hash == content_hash:
        return latest_version
        
    version = DocumentVersion(
        document_id=doc.id,
        content_hash=content_hash,
        extracted_text=text,
        raw_metadata=metadata
    )
    db.add(version)
    await db.commit()
    await db.refresh(version)
    
    # Trigger indexing for MVP
    await index_document_version(db, version.id)
    
    from app.domains.embeddings.pipeline import process_document_version
    await process_document_version(db, version.id)
    
    return version
