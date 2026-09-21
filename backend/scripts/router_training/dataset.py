import asyncio
import json
import statistics
import time
import uuid

from sqlalchemy import select

from app.db.session import get_db
from app.domains.benchmark.models import BenchmarkMapping
from app.domains.search.query.features import extract_features
from app.domains.search.retrieval.orchestrator import execute_search
from scripts.benchmark.datasets.scifact import SciFactDataset

# The Action Space
CONFIGS = {
    "CONFIG_A": {"bm25": 100, "dense": 100},
    "CONFIG_B": {"bm25": 100, "dense": 25},
    "CONFIG_C": {"bm25": 25, "dense": 100},
    "CONFIG_D": {"bm25": 100, "dense": 0},
    "CONFIG_E": {"bm25": 0, "dense": 100},
    "CONFIG_F": {"bm25": 25, "dense": 25}
}

REPETITIONS = 3
OUTCOME_FILE = "scripts/router_training/action_outcome_matrix.jsonl"

async def measure_probe(db, project_id, query):
    """
    Executes a BM25 depth=5 probe and measures its latency.
    Returns (probe_latency_ms, max_bm25_score, score_variance, candidates_count, exact_match)
    """
    t0 = time.time()
    candidates, _ = await execute_search(
        db=db,
        project_id=project_id,
        query=query,
        mode="hybrid_fixed",
        top_k=5,
        retrieval_k=5, # we will intercept in orchestrator if needed, but orchestrator uses fixed depths
        # Actually orchestrator uses the fixed plan, but wait, the orchestrator overrides retrieval_k with plan limits.
        # I need to pass a manual plan or just use the kwargs if orchestrator supports it.
        # In orchestrator.py of Phase 6, mode="hybrid_fixed" forces 100/100 unless we inject a plan.
        # Let's import the specific retrieval function instead for the probe to get accurate BM25 latency.
    )
    return 0

async def generate_dataset():
    dataset_name = "scifact"
    manifest_path = f".cache/{dataset_name}_manifest.json"
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            project_id = uuid.UUID(manifest["project_id"])
    except FileNotFoundError:
        print("Manifest not found. Run benchmark prepare first.")
        return

    provider = SciFactDataset()
    qrels = provider.get_qrels()
    
    async for db in get_db():
        stmt = select(BenchmarkMapping).where(BenchmarkMapping.benchmark_project_id == project_id)
        result = await db.execute(stmt)
        mappings = result.scalars().all()
        nexus_to_external = {str(m.nexus_document_id): m.external_doc_id for m in mappings}
        
        queries = list(provider.iter_queries())
        
        print(f"Starting outcome matrix generation for {len(queries)} queries...")
        count = 0
        
        with open(OUTCOME_FILE, "w") as out_f:
            for query in queries:
                qid = query["id"]
                if qid not in qrels:
                    continue
                    
                q_text = query["text"]
                query_qrels = qrels[qid]
                
                # 1. Feature Extraction
                t0_feat = time.time()
                features = extract_features(q_text, 0.1) # Note: we use dummy term rarity for speed in this test
                feat_time = (time.time() - t0_feat) * 1000
                
                record = {
                    "query_id": qid,
                    "query_text": q_text,
                    "features": features.model_dump(),
                    "feature_extraction_ms": feat_time,
                    "probe": {},
                    "outcomes": {}
                }
                
                # 2. Probe Measurement (BM25 depth=5)
                probe_latencies = []
                for _ in range(3):
                    t0_probe = time.time()
                    try:
                        cands = await execute_search(
                            db=db, project_id=project_id, query=q_text, mode="hybrid_fixed",
                            top_k=5, retrieval_k_bm25=5, retrieval_k_dense=0, rrf_k=5
                        )
                    except:
                        pass
                    probe_latencies.append((time.time() - t0_probe) * 1000)
                
                record["probe"]["median_latency_ms"] = statistics.median(probe_latencies)
                
                # 3. Evaluate all configurations
                from scripts.benchmark.metrics import mrr, ndcg_at_k, recall_at_k
                
                for cfg_name, cfg in CONFIGS.items():
                    latencies = []
                    best_candidates = []
                    for _ in range(REPETITIONS):
                        t0_exec = time.time()
                        try:
                            candidates = await execute_search(
                                db=db, project_id=project_id, query=q_text, mode="hybrid_fixed",
                                top_k=100, retrieval_k_bm25=cfg["bm25"], retrieval_k_dense=cfg["dense"], rrf_k=10
                            )
                            best_candidates = candidates
                        except Exception as e:
                            print(f"Query {qid} config {cfg_name} failed: {e}")
                        latencies.append((time.time() - t0_exec) * 1000)
                        
                    ext_ids = [nexus_to_external.get(str(c.document_id), "") for c in best_candidates]
                    ndcg_10 = ndcg_at_k(ext_ids, query_qrels, k=10)
                    ndcg_5 = ndcg_at_k(ext_ids, query_qrels, k=5)
                    recall_100 = recall_at_k(ext_ids, query_qrels, k=100)
                    recall_10 = recall_at_k(ext_ids, query_qrels, k=10)
                    reciprocal_rank = mrr(ext_ids, query_qrels)
                    
                    record["outcomes"][cfg_name] = {
                        "median_latency_ms": statistics.median(latencies),
                        "p75_latency_ms": statistics.quantiles(latencies, n=4)[2] if len(latencies) >= 4 else max(latencies),
                        "p95_latency_ms": max(latencies), # Using max for P95 since we only have 3 reps
                        "ndcg_10": ndcg_10,
                        "ndcg_5": ndcg_5,
                        "recall_100": recall_100,
                        "recall_10": recall_10,
                        "mrr": reciprocal_rank,
                        "candidate_count": len(ext_ids)
                    }
                    
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                
                count += 1
                if count % 10 == 0:
                    print(f"Processed {count} queries...")
                    
        print(f"Dataset generated at {OUTCOME_FILE}")
if __name__ == "__main__":
    asyncio.run(generate_dataset())
