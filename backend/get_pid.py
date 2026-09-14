import asyncio
import json
from sqlalchemy import text
from app.db.session import engine

async def main():
    try:
        async with engine.connect() as conn:
            # Strictly select the project that has the complete 5,183 documents
            res = await conn.execute(text("""
                SELECT p.id, COUNT(d.id) as doc_count 
                FROM projects p 
                JOIN documents d ON d.project_id = p.id 
                GROUP BY p.id 
                HAVING COUNT(d.id) = 5183 
                ORDER BY p.created_at DESC 
                LIMIT 1
            """))
            row = res.first()
            if not row:
                print("ERROR: No project found with completed SciFact corpus (5,183 documents)!")
                print("Run 'uv run python -m scripts.benchmark prepare --dataset scifact' to ingest first.")
                return

            project_id = row[0]
            manifest = {
                "dataset_name": "scifact",
                "dataset_version": "1.0.0",
                "project_id": str(project_id),
                "corpus_size": 5183,
                "status": "READY"
            }
            with open(".cache/scifact_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            
            print(f"Verified project {project_id} (5,183 documents). Manifest written successfully.")
    except Exception as e:
        print(f"Database connection error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
