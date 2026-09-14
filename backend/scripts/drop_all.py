import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from app.db.base import Base
from app.config.settings import settings

import app.domains.users.models
import app.domains.projects.models
import app.domains.crawler.models
import app.domains.documents.models
import app.domains.search.models
import app.domains.embeddings.models
import app.domains.benchmark.models

async def main():
    engine = create_async_engine(settings.SQLALCHEMY_DATABASE_URI)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE;"))

if __name__ == "__main__":
    asyncio.run(main())
