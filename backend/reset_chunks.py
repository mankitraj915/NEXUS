import asyncio
from sqlalchemy import text
from app.db.session import engine
async def main():
    async with engine.connect() as conn:
        await conn.execute(text("UPDATE document_chunks SET status='PENDING' WHERE status='FAILED'"))
        await conn.commit()
        print("Reset FAILED to PENDING")
asyncio.run(main())
