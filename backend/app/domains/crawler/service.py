import asyncio
import logging
import uuid
from datetime import UTC, datetime
from urllib.parse import urlparse

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.domains.crawler.fetcher import fetch_url
from app.domains.crawler.frontier import (
    dequeue_target,
    enqueue_target,
    mark_target_failed,
    mark_target_success,
)
from app.domains.crawler.models import CrawlJob, CrawlTarget
from app.domains.crawler.normalizer import normalize_url, resolve_url
from app.domains.crawler.parser import parse_html
from app.domains.crawler.schemas import CrawlJobCreate
from app.domains.documents.service import save_document
from app.domains.embeddings.pipeline import process_pending_chunks

logger = logging.getLogger(__name__)


async def create_job(db: AsyncSession, job_in: CrawlJobCreate) -> CrawlJob:
    job = CrawlJob(
        project_id=job_in.project_id,
        seed_url=job_in.seed_url,
        max_pages=job_in.max_pages,
        max_depth=job_in.max_depth,
        respect_robots=job_in.respect_robots,
        concurrency_limit=job_in.concurrency_limit,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    normalized_seed = normalize_url(job.seed_url)
    domain = urlparse(normalized_seed).netloc
    await enqueue_target(db, job.id, normalized_seed, job.seed_url, domain, 0)

    return job


async def get_job_stats(db: AsyncSession, job_id: uuid.UUID) -> dict[str, int | None]:
    stmt_total = select(func.count(CrawlTarget.id)).where(CrawlTarget.job_id == job_id)
    stmt_success = select(func.count(CrawlTarget.id)).where(
        CrawlTarget.job_id == job_id, CrawlTarget.status == "SUCCESS"
    )
    stmt_failed = select(func.count(CrawlTarget.id)).where(
        CrawlTarget.job_id == job_id, CrawlTarget.status == "FAILED"
    )

    total = (await db.execute(stmt_total)).scalar()
    success = (await db.execute(stmt_success)).scalar()
    failed = (await db.execute(stmt_failed)).scalar()

    return {"discovered": total, "fetched": success, "failed": failed}


async def _crawl_worker(job_id: uuid.UUID, semaphore: asyncio.Semaphore) -> None:
    async with AsyncSessionLocal() as db:
        stmt = select(CrawlJob).where(CrawlJob.id == job_id)
        job = (await db.execute(stmt)).scalars().first()
        if not job:
            return

        job.status = "RUNNING"
        job.started_at = datetime.now(UTC)
        await db.commit()

        allowed_domain = urlparse(normalize_url(job.seed_url)).netloc
        pages_crawled = 0

        while pages_crawled < job.max_pages:
            target = await dequeue_target(db, job.id)
            if not target:
                # Basic termination if frontier is exhausted
                break

            async with semaphore:
                result = await fetch_url(target.normalized_url, job.respect_robots)

                if result.status_code == 200 and result.html:
                    await mark_target_success(db, target, result.status_code)

                    parsed = parse_html(result.html, target.normalized_url)

                    canonical = parsed.metadata.get("og:url", target.normalized_url)
                    await save_document(
                        db,
                        job.project_id,
                        canonical,
                        target.domain,
                        parsed.text,
                        # Search results read the title from the version metadata.
                        {"title": parsed.title, **parsed.metadata},
                    )
                    # save_document indexes BM25 but leaves the chunks PENDING; embed
                    # them so dense and hybrid search can find the page too.
                    while await process_pending_chunks(db):
                        pass

                    if target.depth < job.max_depth:
                        for link in parsed.links:
                            norm_link = resolve_url(target.normalized_url, link)
                            link_domain = urlparse(norm_link).netloc

                            if link_domain == allowed_domain:
                                await enqueue_target(
                                    db,
                                    job.id,
                                    norm_link,
                                    link,
                                    link_domain,
                                    target.depth + 1,
                                )

                    pages_crawled += 1
                else:
                    await mark_target_failed(
                        db, target, result.error or f"HTTP {result.status_code}"
                    )

        job.status = "COMPLETED"
        job.completed_at = datetime.now(UTC)
        await db.commit()


def start_crawl_background(job_id: uuid.UUID, concurrency_limit: int = 5) -> None:
    semaphore = asyncio.Semaphore(concurrency_limit)
    asyncio.create_task(_crawl_worker(job_id, semaphore))
