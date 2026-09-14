import asyncio
from app.db.session import get_db
async def main():
    async for db in get_db():
        print("Got db:", db)
asyncio.run(main())
