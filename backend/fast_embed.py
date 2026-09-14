import asyncio
import sys
from app.db.session import get_db
from app.domains.embeddings.pipeline import process_pending_chunks
async def main():
    async for db in get_db():
        while True:
            processed = await process_pending_chunks(db, batch_size=512)
            print("Processed:", processed)
            sys.stdout.flush()
            if processed == 0:
                break
asyncio.run(main())
