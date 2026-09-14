from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.domains.crawler.models import CrawlTarget
import uuid
from typing import Optional

async def enqueue_target(db: AsyncSession, job_id: uuid.UUID, normalized_url: str, original_url: str, domain: str, depth: int):
    stmt = select(CrawlTarget).where(CrawlTarget.job_id == job_id, CrawlTarget.normalized_url == normalized_url)
    result = await db.execute(stmt)
    if result.scalars().first():
        return

    target = CrawlTarget(
        job_id=job_id,
        normalized_url=normalized_url,
        original_url=original_url,
        domain=domain,
        depth=depth,
        status="QUEUED"
    )
    db.add(target)
    await db.commit()

async def dequeue_target(db: AsyncSession, job_id: uuid.UUID) -> Optional[CrawlTarget]:
    stmt = select(CrawlTarget).where(
        CrawlTarget.job_id == job_id, 
        CrawlTarget.status == "QUEUED"
    ).order_by(CrawlTarget.created_at.asc()).limit(1)
    
    result = await db.execute(stmt)
    target = result.scalars().first()
    
    if target:
        target.status = "FETCHING"
        await db.commit()
        await db.refresh(target)
        
    return target

async def mark_target_success(db: AsyncSession, target: CrawlTarget, http_status: int):
    target.status = "SUCCESS"
    target.http_status = http_status
    target.attempts += 1
    await db.commit()

async def mark_target_failed(db: AsyncSession, target: CrawlTarget, error: str):
    target.status = "FAILED"
    target.error_message = error
    target.attempts += 1
    await db.commit()
