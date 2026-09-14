import argparse
import asyncio
import json
import uuid
import time
from datetime import datetime, timezone
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db, engine
from app.db.base import Base
from app.domains.projects.models import Project
from app.domains.documents.service import save_document
from app.domains.documents.models import Document, DocumentVersion
from app.domains.embeddings.models import Chunk, ChunkEmbedding
from app.domains.benchmark.models import BenchmarkMapping
from app.domains.embeddings.pipeline import process_pending_chunks
from scripts.benchmark.datasets.scifact import SciFactDataset

async def create_benchmark_project(db: AsyncSession, name: str) -> Project:
    # Use a dummy user or fetch first user. For simplicity, we just create a dummy user.
    # Wait, in the actual system we need a User. Let's fetch the first user, or create one if none exists.
    from app.domains.users.models import User
    stmt = select(User)
    result = await db.execute(stmt)
    user = result.scalars().first()
    if not user:
        user = User(email="benchmark@nexus.com", hashed_password="fake")
        db.add(user)
        await db.flush()

    project = Project(name=name, user_id=user.id)
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project

async def cmd_prepare(args):
    dataset_name = args.dataset
    if dataset_name != "scifact":
        raise ValueError(f"Dataset {dataset_name} not supported yet.")
    
    provider = SciFactDataset()
    provider.prepare()
    
    manifest_path = f".cache/{provider.get_name()}_manifest.json"
    
    if not args.force_prepare:
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                if manifest.get("status") == "READY":
                    print(f"Benchmark corpus is already READY for project: {manifest['project_id']}")
                    return
        except FileNotFoundError:
            pass

    async for db in get_db():
        print(f"Creating Benchmark Project for {provider.get_name()} {provider.get_version()}...")
        project_name = f"Benchmark: {provider.get_name()} {provider.get_version()} {int(time.time())}"
        project = await create_benchmark_project(db, project_name)
        
        print(f"Ingesting corpus...")
        count = 0
        for item in provider.iter_corpus():
            # Combine title and text for search
            full_text = f"{item['title']}\n\n{item['text']}" if item['title'] else item['text']
            
            # Using save_document which handles document creation and BM25 indexing
            # Note: save_document expects url, domain, text, title
            canonical_url = f"benchmark://{provider.get_name()}/{item['id']}"
            doc_version = await save_document(
                db=db,
                project_id=project.id,
                canonical_url=canonical_url,
                source_domain=provider.get_name(),
                text=full_text,
                metadata={"external_id": item['id'], "title": item['title']}
            )
            
            # Save explicit mapping
            mapping = BenchmarkMapping(
                dataset_name=provider.get_name(),
                dataset_version=provider.get_version(),
                external_doc_id=item['id'],
                nexus_document_id=doc_version.document_id,
                benchmark_project_id=project.id
            )
            db.add(mapping)
            count += 1
            if count % 500 == 0:
                print(f"Ingested {count} documents...")
                await db.commit()
        
        await db.commit()
        print(f"Finished ingesting {count} documents. Now chunking and embedding...")
        
        # Now run process_pending_chunks
        while True:
            processed = await process_pending_chunks(db, batch_size=512)
            if processed == 0:
                break
        print("Embeddings generation complete.")
        
        # Verify indexing completeness
        # 1. Total mapping count == corpus count
        stmt = select(BenchmarkMapping).where(BenchmarkMapping.benchmark_project_id == project.id)
        result = await db.execute(stmt)
        mappings = result.scalars().all()
        assert len(mappings) == count, f"Mapping count {len(mappings)} != corpus count {count}"
        
        # 2. Check that all documents have at least one chunk embedding
        # Actually a simpler check is whether all chunks are processed
        # A full check would be heavy, but we know process_pending_chunks loops until done.
        
        manifest = {
            "dataset_name": provider.get_name(),
            "dataset_version": provider.get_version(),
            "project_id": str(project.id),
            "corpus_size": count,
            "status": "READY"
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"Preparation complete. Manifest written to {manifest_path}")

async def cmd_evaluate(args):
    import os
    dataset_name = args.dataset
    split = getattr(args, "split", "test")
    manifest_path = f".cache/{dataset_name}_manifest.json"
    
    if dataset_name != "scifact":
        raise ValueError(f"Dataset '{dataset_name}' not supported.")
        
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            project_id = uuid.UUID(manifest["project_id"])
    except FileNotFoundError:
        raise RuntimeError(f"Manifest not found at {manifest_path}. Run prepare first.")
        
    provider = SciFactDataset()
    qrels = provider.get_qrels(split=split)
    queries = list(provider.iter_queries(split=split))
    
    # Pre-flight query count assertion
    expected_queries = 300 if split == "test" else 809 if split == "train" else None
    if expected_queries and len(queries) != expected_queries:
        raise ValueError(
            f"Pre-flight query validation failed for split '{split}': "
            f"expected {expected_queries} queries, found {len(queries)}"
        )
    print(f"Pre-flight query check passed: {len(queries)} queries loaded for split='{split}'.")
    
    from app.domains.search.retrieval.orchestrator import execute_search
    from scripts.benchmark.experiment import ExperimentTracker
    
    async for db in get_db():
        # Pre-flight project resolution & completeness check
        stmt_proj = select(Project).where(Project.id == project_id)
        project = (await db.execute(stmt_proj)).scalars().first()
        if not project:
            raise RuntimeError(f"Target benchmark project {project_id} does not exist in database.")

        stmt_count = select(func.count(Document.id)).where(Document.project_id == project_id)
        doc_count = (await db.execute(stmt_count)).scalar() or 0
        expected_corpus = 5183 if dataset_name == "scifact" else None
        if expected_corpus and doc_count != expected_corpus:
            raise RuntimeError(
                f"Corpus verification failed: project {project_id} contains {doc_count} documents, "
                f"expected {expected_corpus}. Incomplete ingestion detected - aborting evaluation."
            )

        stmt = select(BenchmarkMapping).where(BenchmarkMapping.benchmark_project_id == project_id)
        result = await db.execute(stmt)
        mappings = result.scalars().all()
        if expected_corpus and len(mappings) != expected_corpus:
            raise RuntimeError(
                f"BenchmarkMapping count mismatch for project {project_id}: "
                f"expected {expected_corpus}, found {len(mappings)}."
            )
        print(f"Pre-flight database check passed: Project '{project.name}' verified with {doc_count} documents.")
        nexus_to_external = {str(m.nexus_document_id): m.external_doc_id for m in mappings}
        
        experiments = [
            # 1. Lexical (BM25)
            {"name": "BM25 (lexical)", "mode": "lexical", "retrieval_k_bm25": 100, "retrieval_k_dense": 0, "rrf_k": 60, "top_k": 100},
            
            # 2. Dense
            {"name": "Dense (dense)", "mode": "dense", "retrieval_k_bm25": 0, "retrieval_k_dense": 100, "rrf_k": 60, "top_k": 100},
            
            # 3. Fixed Production Baseline (100/100, k=10)
            {"name": "Fixed Baseline (100/100, k=10)", "mode": "hybrid_fixed", "retrieval_k_bm25": 100, "retrieval_k_dense": 100, "rrf_k": 10, "top_k": 100},
        ]
        
        results_map = {}
        
        for exp in experiments:
            print(f"\nRunning experiment: {exp['name']} [{split} split: {len(queries)} queries]")
            tracker = ExperimentTracker()
            count = 0
            total_candidates_sum = 0
            
            for query in queries:
                qid = query["id"]
                if qid not in qrels:
                    continue
                query_qrels = qrels[qid]
                
                t0 = time.time()
                candidates = await execute_search(
                    db=db,
                    project_id=project_id,
                    query=query["text"],
                    mode=exp["mode"],
                    top_k=exp["top_k"],
                    retrieval_k_bm25=exp["retrieval_k_bm25"],
                    retrieval_k_dense=exp["retrieval_k_dense"],
                    rerank_k=0,
                    rrf_k=exp["rrf_k"],
                    return_analysis=False
                )
                
                total_candidates_sum += len(candidates)
                latency = (time.time() - t0) * 1000
                tracker.add_latency(latency)
                
                ext_ids = [nexus_to_external.get(str(c.document_id), "") for c in candidates]
                tracker.add_metrics(ext_ids, query_qrels)
                
                count += 1
                if count % 100 == 0:
                    print(f"Evaluated {count}/{len(queries)} queries...")
                    
            summary = tracker.get_summary()
            summary["avg_returned_candidates"] = total_candidates_sum / max(1, count)
            results_map[exp['name']] = summary
            print(f"[{exp['name']}] NDCG@10: {summary['ndcg@10']:.4f} | Recall@100: {summary['recall@100']:.4f} | Latency P95: {summary['latency_p95']:.2f}ms")
            
        os.makedirs("docs", exist_ok=True)
        report_file = f"docs/benchmark-evaluation-{split}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# NEXUS Benchmark Evaluation — {dataset_name.upper()} ({split.upper()} split)\n\n")
            f.write(f"- **Corpus**: {expected_corpus} documents\n")
            f.write(f"- **Queries**: {len(queries)} queries\n")
            f.write(f"- **Project ID**: `{project_id}`\n\n")
            f.write("| Configuration | NDCG@10 | Recall@100 | Avg Candidates | Latency Avg | Latency P95 |\n")
            f.write("|---|---|---|---|---|---|\n")
            
            for exp in experiments:
                s = results_map[exp['name']]
                f.write(f"| {exp['name']} | {s['ndcg@10']:.4f} | {s['recall@100']:.4f} | {s['avg_returned_candidates']:.1f} | {s['latency_avg']:.2f}ms | {s['latency_p95']:.2f}ms |\n")
                
        print(f"\nEvaluation complete. Full report written to {report_file}.")

def main():
    parser = argparse.ArgumentParser(description="NEXUS Benchmark Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_prepare = subparsers.add_parser("prepare")
    p_prepare.add_argument("--dataset", required=True, choices=["scifact"])
    p_prepare.add_argument("--force-prepare", action="store_true")
    
    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--dataset", required=True, choices=["scifact"])
    p_eval.add_argument("--split", required=True, choices=["test", "train"], help="Explicit split to evaluate: 'test' (canonical 300 queries) or 'train' (809 queries)")
    
    p_gen = subparsers.add_parser("generate-llm-variants")
    p_gen.add_argument("--dataset", required=True, choices=["scifact"])
    p_gen.add_argument("--preflight", action="store_true", help="Run config validation and cache integrity check, no real API calls")
    p_gen.add_argument("--smoke-test", action="store_true", help="Run exactly one real API call to test the pipeline")
    
    p_eval_llm = subparsers.add_parser("evaluate-llm")
    p_eval_llm.add_argument("--dataset", required=True, choices=["scifact"])
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        asyncio.run(cmd_prepare(args))
    elif args.command == "evaluate":
        asyncio.run(cmd_evaluate(args))
    elif args.command == "generate-llm-variants":
        from scripts.benchmark.llm_eval import cmd_generate_llm_variants
        asyncio.run(cmd_generate_llm_variants(args))
    elif args.command == "evaluate-llm":
        from scripts.benchmark.llm_eval import cmd_evaluate_llm
        asyncio.run(cmd_evaluate_llm(args))

if __name__ == "__main__":
    main()
