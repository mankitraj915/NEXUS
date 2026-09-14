import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.domains.crawler import schemas, service
from app.domains.crawler.models import CrawlJob

router = APIRouter()


@router.post(
    "/jobs",
    response_model=schemas.CrawlJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_crawl_job(
    job_in: schemas.CrawlJobCreate, db: AsyncSession = Depends(get_db)
):
    job = await service.create_job(db, job_in)
    service.start_crawl_background(job.id, job.concurrency_limit)
    return job


@router.get("/jobs/{job_id}", response_model=schemas.CrawlJobResponse)
async def get_crawl_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(CrawlJob).where(CrawlJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalars().first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/stats")
async def get_crawl_stats(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    return await service.get_job_stats(db, job_id)
