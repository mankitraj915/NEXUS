

import asyncio
import uuid
import time
import json
import os
from sqlalchemy import select
from app.db.session import get_db
from app.domains.benchmark.models import BenchmarkMapping
from scripts.benchmark.datasets.scifact import SciFactDataset
from app.domains.search.query.llm_provider import OpenAICompatibleProvider
from app.domains.search.query.drift import SemanticDriftFilter
from app.domains.search.query.cache import VariantCache
from app.domains.search.query.transformer import LLMQueryTransformer
from scripts.benchmark.experiment import ExperimentTracker
from app.domains.search.retrieval.orchestrator import execute_search

async def cmd_generate_llm_variants(args):
    dataset_name = args.dataset
    if dataset_name != "scifact":
        raise ValueError("Only scifact supported.")
        
    try:
        provider = OpenAICompatibleProvider()
    except ValueError as e:
        print(f"Error: {e}")
        return
        
    cache = VariantCache()
    
    if args.preflight:
        print("Pre-flight successful: Configuration validated and cache initialized.")
        return
        
    ds = SciFactDataset()
    queries = list(ds.iter_queries())
    
    max_variants = 3
    drift_filter = SemanticDriftFilter(threshold=0.70)
    transformer = LLMQueryTransformer(provider, drift_filter, cache, max_variants)
    
    if args.smoke_test:
        print("Running smoke-test on a single query...")
        test_query = queries[0]["text"]
        test_variants = await transformer.transform(test_query)
        if not test_variants or len(test_variants) == 1:
            print("Warning: Single test query yielded no new variants. Proceeding with caution.")
        else:
            print(f"Smoke-test successful. Generated {len(test_variants)-1} accepted variants.")
        return

    
    print(f"Starting offline LLM variant generation for {len(queries)} queries...")
    
    count = 0
    total_variants_generated = 0
    total_variants_accepted = 0
    total_latency = 0.0
    
    # Gather metrics
    for q in queries:
        t0 = time.time()
        variants = await transformer.transform(q["text"])
        lat = time.time() - t0
        total_latency += lat
        
        # we can't easily track raw generated vs rejected unless we inspect the cache
        # Let's peek at the cache for metrics
        cache_key = cache.generate_key(q["text"], transformer._get_config_dict())
        c_data = cache.get(cache_key)
        if c_data:
            total_variants_generated += len(c_data.get("variants", []))
            total_variants_accepted += sum(1 for v in c_data.get("variants", []) if v.get("is_accepted"))
            
        count += 1
        if count % 10 == 0:
            print(f"Processed {count} / {len(queries)}...")
            
    avg_lat = total_latency / max(1, count)
    avg_vars = total_variants_accepted / max(1, count)
    print(f"--- Generation Complete ---")
    print(f"Total queries processed: {count}")
    print(f"Total variants generated: {total_variants_generated}")
    print(f"Total variants accepted: {total_variants_accepted}")
    print(f"Total variants rejected: {total_variants_generated - total_variants_accepted}")
    print(f"Average accepted variants/query: {avg_vars:.2f}")
    print(f"Average generation latency per query: {avg_lat*1000:.2f}ms")
    
    # Write generation manifest
    manifest_path = f".cache/{dataset_name}_llm_manifest.json"
    manifest = {
        "dataset_version": ds.get_version(),
        "provider": provider.provider_name,
        "model": provider.model_name,
        "model_version": "unknown",
        "prompt_version": transformer.version,
        "temperature": provider.temperature,
        "max_variants": max_variants,
        "drift_threshold": drift_filter.threshold,
        "timestamp": time.time()
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Manifest written to {manifest_path}")



async def cmd_evaluate_llm(args):
    dataset_name = args.dataset
    manifest_path = f".cache/{dataset_name}_manifest.json"
    llm_manifest_path = f".cache/{dataset_name}_llm_manifest.json"
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            project_id = uuid.UUID(manifest["project_id"])
    except FileNotFoundError:
        print("Dataset Manifest not found. Run prepare first.")
        return
        
    try:
        with open(llm_manifest_path, 'r') as f:
            llm_manifest = json.load(f)
    except FileNotFoundError:
        print("LLM generation manifest not found. Run generate-llm-variants first.")
        return

    ds = SciFactDataset()
    qrels = ds.get_qrels()
    queries = list(ds.iter_queries())

    experiments = [
        {"name": "A. Fixed RRF", "mode": "hybrid_fixed", "rrf_k": 10},
        {"name": "B. Rule-Based Multi-Query (Strategy A)", "mode": "multi_query_rule", "strategy": "strategy_a"},
        {"name": "C. LLM (1 variant / weight 1.0)", "mode": "multi_query_llm", "max_variants": 1, "variant_weight": 1.0},
        {"name": "D. LLM (2 variants / weight 1.0)", "mode": "multi_query_llm", "max_variants": 2, "variant_weight": 1.0},
        {"name": "E. LLM (3 variants / weight 1.0)", "mode": "multi_query_llm", "max_variants": 3, "variant_weight": 1.0},
        {"name": "F. LLM (2 variants / weight 0.75)", "mode": "multi_query_llm", "max_variants": 2, "variant_weight": 0.75},
    ]

    results_map = {}
    
    async for db in get_db():
        stmt = select(BenchmarkMapping).where(BenchmarkMapping.benchmark_project_id == project_id)
        result = await db.execute(stmt)
        mappings = result.scalars().all()
        nexus_to_external = {str(m.nexus_document_id): m.external_doc_id for m in mappings}
        
        for exp in experiments:
            print(f"\nRunning experiment: {exp['name']}")
            tracker = ExperimentTracker()
            count = 0
            
            for query in queries:
                qid = query["id"]
                if qid not in qrels:
                    continue
                query_qrels = qrels[qid]
                
                t0 = time.time()
                # Need to update execute_search to handle LLM mode
                candidates = await execute_search(
                    db=db,
                    project_id=project_id,
                    query=query["text"],
                    mode=exp["mode"],
                    top_k=100,
                    retrieval_k=100,
                    rerank_k=0,
                    rrf_k=10,
                    multi_query_strategy=exp.get("strategy", "strategy_a"),
                    llm_max_variants=exp.get("max_variants", 3),
                    llm_variant_weight=exp.get("variant_weight", 1.0)
                )
                
                latency = (time.time() - t0) * 1000
                tracker.add_latency(latency)
                
                ext_ids = [nexus_to_external.get(str(c.document_id), "") for c in candidates]
                tracker.add_metrics(ext_ids, query_qrels)
                
                count += 1
                if count % 100 == 0:
                    print(f"Evaluated {count} queries...")
                    
            summary = tracker.get_summary()
            results_map[exp['name']] = summary
            print(f"NDCG@10: {summary['ndcg@10']:.4f} | Recall@100: {summary['recall@100']:.4f} | Latency P95: {summary['latency_p95']:.2f}ms")

        # Also evaluate Drift Thresholds
        print("\nEvaluating Drift Thresholds (using 3 variants / weight 1.0)")
        thresholds = [0.70, 0.75, 0.80, 0.85]
        for t in thresholds:
            name = f"Drift Threshold {t:.2f}"
            print(f"Running experiment: {name}")
            tracker = ExperimentTracker()
            count = 0
            for query in queries:
                qid = query["id"]
                if qid not in qrels: continue
                query_qrels = qrels[qid]
                
                candidates = await execute_search(
                    db=db,
                    project_id=project_id,
                    query=query["text"],
                    mode="multi_query_llm",
                    top_k=100,
                    retrieval_k=100,
                    rerank_k=0,
                    rrf_k=10,
                    multi_query_strategy="strategy_a",
                    llm_max_variants=3,
                    llm_variant_weight=1.0,
                    llm_drift_threshold=t
                )
                
                ext_ids = [nexus_to_external.get(str(c.document_id), "") for c in candidates]
                tracker.add_metrics(ext_ids, query_qrels)
                count += 1
                
            summary = tracker.get_summary()
            results_map[name] = summary
            print(f"NDCG@10: {summary['ndcg@10']:.4f} | Recall@100: {summary['recall@100']:.4f}")

        with open("docs/llm-query-transformation-evaluation.md", "w", encoding="utf-8") as f:
            f.write("# NEXUS Phase 8B Query Transformation Evaluation\n\n")
            f.write("## 1. Quality / Latency Frontier\n")
            f.write("| Configuration | NDCG@10 | Recall@100 | Latency P95 |\n")
            f.write("|---|---|---|---|\n")
            
            for exp_name, s in results_map.items():
                lat = f"{s.get('latency_p95', 0):.2f}ms" if s.get('latency_p95') else "N/A"
                f.write(f"| {exp_name} | {s['ndcg@10']:.4f} | {s['recall@100']:.4f} | {lat} |\n")
                
        print("\nEvaluation complete. Full report written to docs/llm-query-transformation-evaluation.md.")
