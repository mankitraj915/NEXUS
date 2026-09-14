import asyncio
from sqlalchemy import text
from app.db.session import engine
async def main():
    async with engine.connect() as conn:
        res = await conn.execute(text("SELECT count(*) FROM document_chunks WHERE status='EMBEDDED'"))
        print("Embedded:", res.scalar())
        res = await conn.execute(text("SELECT count(*) FROM document_chunks WHERE status='PENDING'"))
        print("Pending:", res.scalar())
asyncio.run(main())
