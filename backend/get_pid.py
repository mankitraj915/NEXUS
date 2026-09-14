import asyncio
import json
import time
from sqlalchemy import text
from app.db.session import engine

async def main():
    while True:
        try:
            async with engine.connect() as conn:
                res = await conn.execute(text("SELECT id FROM projects ORDER BY created_at DESC LIMIT 1"))
                project_id = res.scalar()
                
                manifest = {
                    "dataset_name": "scifact",
                    "dataset_version": "1.0.0",
                    "project_id": str(project_id),
                    "corpus_size": 5183,
                    "status": "READY"
                }
                with open(".cache/scifact_manifest.json", 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                print("Manifest written:", manifest)
                break
        except Exception as e:
            print("Waiting for db...")
            await asyncio.sleep(2)

asyncio.run(main())
