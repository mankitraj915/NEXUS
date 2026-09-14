from pydantic import BaseModel, ConfigDict
import uuid
from typing import Optional
from datetime import datetime

class CrawlJobBase(BaseModel):
    seed_url: str
    max_pages: int = 100
    max_depth: int = 3
    respect_robots: bool = True
    concurrency_limit: int = 5

class CrawlJobCreate(CrawlJobBase):
    project_id: uuid.UUID

class CrawlJobResponse(CrawlJobBase):
    id: uuid.UUID
    project_id: uuid.UUID
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]

    model_config = ConfigDict(from_attributes=True)
