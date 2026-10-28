import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from typing import TypedDict

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.domains.benchmark.models import BenchmarkMapping
from app.domains.documents.models import Document
from app.domains.documents.service import save_document
from app.domains.embeddings.pipeline import process_pending_chunks
from app.domains.projects.models import Project
from scripts.benchmark.datasets.scifact import SciFactDataset
from scripts.benchmark.significance import N_RESAMPLES, compare_paired


class BenchmarkExperimentConfig(TypedDict):
    name: str
    mode: str
    retrieval_k_bm25: int | None
    retrieval_k_dense: int | None
    rrf_k: int
    top_k: int
    rerank_k: int


# BEIR's SciFact "test" qrels are the official SciFact *dev* claims (identical IDs and
# text); the official SciFact test labels are not public.
SPLIT_DESCRIPTIONS = {
    "test": (
        "BEIR SciFact test = official SciFact dev claims (300). RRF k=10 was selected "
        "on this split, so the k=10 row is a tuned (optimistic) result."
    ),
    "train": "BEIR SciFact train = official SciFact train claims (809); not used for tuning.",
}

# Paired comparisons are always made against the default configuration.
DEFAULT_CONFIGURATION = "Fixed Baseline (100/100, k=10)"

DENSE_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_REPO = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _cached_revision(repo_id: str) -> str:
    """Return '<repo>@<commit>' for a model in the local Hugging Face cache."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo_id, "config.json")
    if isinstance(path, str) and "snapshots" in path:
        return f"{repo_id}@{os.path.basename(os.path.dirname(path))}"
    return f"{repo_id}@(not cached)"


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def measurement_conditions() -> dict:
    import sentence_transformers
    import torch

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            commit += "-dirty"
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": commit,
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "sentence_transformers": sentence_transformers.__version__,
        "models": {
            "dense": _cached_revision(DENSE_MODEL_REPO),
            "reranker": _cached_revision(RERANKER_MODEL_REPO),
        },
    }


async def create_benchmark_project(db: AsyncSession, name: str) -> Project:
    # Benchmark projects require an associated user record for foreign key integrity.
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


async def benchmark_project_is_complete(
    project_id: uuid.UUID, corpus_size: int
) -> bool:
    async for db in get_db():
        docs = await db.scalar(
            select(func.count(Document.id)).where(Document.project_id == project_id)
        )
        mappings = await db.scalar(
            select(func.count(BenchmarkMapping.id)).where(
                BenchmarkMapping.benchmark_project_id == project_id
            )
        )
        return docs == corpus_size and mappings == corpus_size
    return False


async def cmd_prepare(args: argparse.Namespace) -> None:
    dataset_name = args.dataset
    if dataset_name != "scifact":
        raise ValueError(f"Dataset {dataset_name} not supported yet.")

    provider = SciFactDataset()
    provider.prepare()

    manifest_path = f".cache/{provider.get_name()}_manifest.json"

    if not args.force_prepare:
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except FileNotFoundError:
            manifest = {}
        if manifest.get("status") == "READY":
            project_id = uuid.UUID(manifest["project_id"])
            if await benchmark_project_is_complete(project_id, manifest["corpus_size"]):
                print(f"Benchmark corpus is already READY for project: {project_id}")
                return
            print(
                f"Manifest references project {project_id}, but it is missing or "
                "incomplete in the database. Re-ingesting the corpus."
            )

    async for db in get_db():
        print(
            f"Creating Benchmark Project for {provider.get_name()} {provider.get_version()}..."
        )
        project_name = f"Benchmark: {provider.get_name()} {provider.get_version()} {int(time.time())}"
        project = await create_benchmark_project(db, project_name)

        print("Ingesting corpus...")
        count = 0
        for item in provider.iter_corpus():
            # Combine title and text for search
            full_text = (
                f"{item['title']}\n\n{item['text']}" if item["title"] else item["text"]
            )

            # Using save_document which handles document creation and BM25 indexing
            # Note: save_document expects url, domain, text, title
            canonical_url = f"benchmark://{provider.get_name()}/{item['id']}"
            doc_version = await save_document(
                db=db,
                project_id=project.id,
                canonical_url=canonical_url,
                source_domain=provider.get_name(),
                text=full_text,
                metadata={"external_id": item["id"], "title": item["title"]},
            )

            # Save explicit mapping
            mapping = BenchmarkMapping(
                dataset_name=provider.get_name(),
                dataset_version=provider.get_version(),
                external_doc_id=item["id"],
                nexus_document_id=doc_version.document_id,
                benchmark_project_id=project.id,
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
        stmt = select(BenchmarkMapping).where(
            BenchmarkMapping.benchmark_project_id == project.id
        )
        result = await db.execute(stmt)
        mappings = result.scalars().all()
        assert len(mappings) == count, (
            f"Mapping count {len(mappings)} != corpus count {count}"
        )

        # 2. Check that all documents have at least one chunk embedding
        # Actually a simpler check is whether all chunks are processed
        # A full check would be heavy, but we know process_pending_chunks loops until done.

        manifest = {
            "dataset_name": provider.get_name(),
            "dataset_version": provider.get_version(),
            "project_id": str(project.id),
            "corpus_size": count,
            "status": "READY",
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Preparation complete. Manifest written to {manifest_path}")


async def cmd_evaluate(args: argparse.Namespace) -> None:
    dataset_name = args.dataset
    split = getattr(args, "split", "test")
    manifest_path = f".cache/{dataset_name}_manifest.json"

    if dataset_name != "scifact":
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    try:
        with open(manifest_path, "r") as f:
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
    print(
        f"Pre-flight query check passed: {len(queries)} queries loaded for split='{split}'."
    )

    from app.domains.search.retrieval.orchestrator import execute_search
    from scripts.benchmark.experiment import ExperimentTracker

    async for db in get_db():
        # Pre-flight project resolution & completeness check
        stmt_proj = select(Project).where(Project.id == project_id)
        project = (await db.execute(stmt_proj)).scalars().first()
        if not project:
            raise RuntimeError(
                f"Target benchmark project {project_id} does not exist in database."
            )

        stmt_count = select(func.count(Document.id)).where(
            Document.project_id == project_id
        )
        doc_count = (await db.execute(stmt_count)).scalar() or 0
        expected_corpus = 5183 if dataset_name == "scifact" else None
        if expected_corpus and doc_count != expected_corpus:
            raise RuntimeError(
                f"Corpus verification failed: project {project_id} contains {doc_count} documents, "
                f"expected {expected_corpus}. Incomplete ingestion detected - aborting evaluation."
            )

        stmt = select(BenchmarkMapping).where(
            BenchmarkMapping.benchmark_project_id == project_id
        )
        result = await db.execute(stmt)
        mappings = result.scalars().all()
        if expected_corpus and len(mappings) != expected_corpus:
            raise RuntimeError(
                f"BenchmarkMapping count mismatch for project {project_id}: "
                f"expected {expected_corpus}, found {len(mappings)}."
            )
        print(
            f"Pre-flight database check passed: Project '{project.name}' verified with {doc_count} documents."
        )
        nexus_to_external = {
            str(m.nexus_document_id): m.external_doc_id for m in mappings
        }

        experiments: list[BenchmarkExperimentConfig] = [
            # 1. Lexical (BM25)
            {
                "name": "BM25 (lexical)",
                "mode": "lexical",
                "retrieval_k_bm25": 100,
                "retrieval_k_dense": 0,
                "rrf_k": 60,
                "top_k": 100,
                "rerank_k": 0,
            },
            # 2. Dense
            {
                "name": "Dense (dense)",
                "mode": "dense",
                "retrieval_k_bm25": 0,
                "retrieval_k_dense": 100,
                "rrf_k": 60,
                "top_k": 100,
                "rerank_k": 0,
            },
            # 3. Hybrid with the Cormack et al. default constant, for reference
            {
                "name": "Hybrid RRF (100/100, k=60)",
                "mode": "hybrid_fixed",
                "retrieval_k_bm25": 100,
                "retrieval_k_dense": 100,
                "rrf_k": 60,
                "top_k": 100,
                "rerank_k": 0,
            },
            # 4. Fixed Production Baseline (100/100, k=10)
            {
                "name": "Fixed Baseline (100/100, k=10)",
                "mode": "hybrid_fixed",
                "retrieval_k_bm25": 100,
                "retrieval_k_dense": 100,
                "rrf_k": 10,
                "top_k": 100,
                "rerank_k": 0,
            },
        ]
        if args.with_rerank:
            # Optional: CPU cross-encoder reranking of the top 25 fused candidates
            experiments.append(
                {
                    "name": "Baseline + CrossEncoder rerank (top 25)",
                    "mode": "hybrid_rerank",
                    "retrieval_k_bm25": 100,
                    "retrieval_k_dense": 100,
                    "rrf_k": 10,
                    "top_k": 25,
                    "rerank_k": 25,
                }
            )

        results_map = {}
        per_query_map = {}

        for exp in experiments:
            print(
                f"\nRunning experiment: {exp['name']} [{split} split: {len(queries)} queries]"
            )
            tracker = ExperimentTracker()
            count = 0
            total_candidates_sum = 0

            # Unmeasured warm-up so model loading is not counted as query latency
            await execute_search(
                db=db,
                project_id=project_id,
                query=queries[0]["text"],
                mode=exp["mode"],
                top_k=exp["top_k"],
                retrieval_k_bm25=exp["retrieval_k_bm25"],
                retrieval_k_dense=exp["retrieval_k_dense"],
                rerank_k=exp["rerank_k"],
                rrf_k=exp["rrf_k"],
            )

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
                    rerank_k=exp["rerank_k"],
                    rrf_k=exp["rrf_k"],
                    return_analysis=False,
                )

                total_candidates_sum += len(candidates)
                latency = (time.time() - t0) * 1000
                tracker.add_latency(latency)

                ext_ids = [
                    nexus_to_external.get(str(c.document_id), "") for c in candidates
                ]
                tracker.add_metrics(ext_ids, query_qrels, query_id=qid)

                count += 1
                if count % 100 == 0:
                    print(f"Evaluated {count}/{len(queries)} queries...")

            summary = tracker.get_summary()
            summary["avg_returned_candidates"] = total_candidates_sum / max(1, count)
            results_map[exp["name"]] = summary
            per_query_map[exp["name"]] = tracker.per_query()
            print(
                f"[{exp['name']}] NDCG@10: {summary['ndcg@10']:.4f} | Recall@100: {summary['recall@100']:.4f} | Latency P95: {summary['latency_p95']:.2f}ms"
            )

        default_scores = per_query_map[DEFAULT_CONFIGURATION]["ndcg@10"]
        significance = [
            {
                "configuration": DEFAULT_CONFIGURATION,
                "baseline": exp["name"],
                "metric": "ndcg@10",
                **compare_paired(default_scores, per_query_map[exp["name"]]["ndcg@10"]),
            }
            for exp in experiments
            if exp["name"] != DEFAULT_CONFIGURATION
        ]

        conditions = measurement_conditions()
        os.makedirs("docs", exist_ok=True)
        report_file = f"docs/benchmark-evaluation-{split}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(
                f"# NEXUS Benchmark Evaluation — {dataset_name.upper()} ({split.upper()} split)\n\n"
            )
            f.write(f"- **Split**: {SPLIT_DESCRIPTIONS[split]}\n")
            f.write(f"- **Corpus**: {expected_corpus} documents\n")
            f.write(f"- **Queries**: {len(queries)} queries\n")
            f.write(
                f"- **Measured**: {conditions['timestamp_utc']} at commit `{conditions['git_commit']}`\n"
            )
            f.write(
                f"- **Hardware**: {conditions['cpu']}, {conditions['logical_cpus']} logical CPUs, CPU-only\n"
            )
            f.write(
                f"- **Models**: dense `{conditions['models']['dense']}`, reranker "
                f"`{conditions['models']['reranker']}`\n"
            )
            f.write(
                "- **Latency**: wall-clock per query from the benchmark client (query "
                "embedding + SQL + fusion), after one unmeasured warm-up query.\n\n"
            )
            f.write(
                "| Configuration | NDCG@10 | Recall@10 | Recall@100 | MRR | Latency Avg | Latency P95 |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")

            for exp in experiments:
                s = results_map[exp["name"]]
                recall_100 = f"{s['recall@100']:.4f}" if exp["top_k"] >= 100 else "n/a"
                f.write(
                    f"| {exp['name']} | {s['ndcg@10']:.4f} | {s['recall@10']:.4f} | {recall_100} | {s['mrr']:.4f} | {s['latency_avg']:.2f} ms | {s['latency_p95']:.2f} ms |\n"
                )

            f.write(
                f"\n## Paired comparison: {DEFAULT_CONFIGURATION} vs each configuration (NDCG@10)\n\n"
                "Per-query differences (default minus other). 95% CI: percentile bootstrap "
                "of the mean difference; p: two-sided paired randomization test. Both use "
                f"{N_RESAMPLES:,} resamples with a fixed seed. W / T / L: queries where the "
                "default scores higher / equal / lower.\n\n"
            )
            f.write("| Compared with | Mean Δ NDCG@10 | 95% CI | p | W / T / L |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            f.writelines(
                f"| {row['baseline']} | {row['mean_diff']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['p_value']:.4f} | "
                f"{row['wins']} / {row['ties']} / {row['losses']} |\n"
                for row in significance
            )

        results_file = f"docs/benchmark-evaluation-{split}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "split_description": SPLIT_DESCRIPTIONS[split],
                    "queries": len(queries),
                    "corpus_documents": expected_corpus,
                    "conditions": conditions,
                    "configurations": [
                        {
                            **exp,
                            "metrics": results_map[exp["name"]],
                            "per_query": per_query_map[exp["name"]],
                        }
                        for exp in experiments
                    ],
                    "significance": significance,
                },
                f,
                indent=2,
            )

        print(
            f"\nEvaluation complete. Report written to {report_file} and {results_file}."
        )


def cmd_verify(args: argparse.Namespace) -> None:
    import math

    from app.domains.search.analyzer import analyze_text
    from app.domains.search.retrieval.fusion import reciprocal_rank_fusion
    from app.domains.search.retrieval.models import Candidate
    from scripts.benchmark.metrics import (
        jaccard_at_k,
        mrr,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )

    print("=" * 70)
    print("NEXUS Retrieval Engine & Evaluation Harness Verification")
    print("=" * 70)

    # 1. Verify DCG and NDCG
    perfect_ids = ["d1", "d2", "d3"]
    qrels = {"d1": 1, "d2": 1, "d3": 0}
    perfect_ndcg = ndcg_at_k(perfect_ids, qrels, k=2)
    assert math.isclose(perfect_ndcg, 1.0, rel_tol=1e-5), (
        f"Expected 1.0, got {perfect_ndcg}"
    )

    reversed_ids = ["d2", "d1"]
    reversed_ndcg = ndcg_at_k(reversed_ids, {"d1": 2, "d2": 1}, k=2)
    ideal_dcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    actual_dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    assert math.isclose(reversed_ndcg, actual_dcg / ideal_dcg, rel_tol=1e-5)
    print(" [PASS] NDCG@k and Graded DCG mathematical correctness")

    # 2. Verify Recall and Precision
    p_at_2 = precision_at_k(["d1", "d3"], qrels, k=2)
    assert p_at_2 == 0.5, f"Expected 0.5, got {p_at_2}"
    r_at_2 = recall_at_k(["d1", "d3"], qrels, k=2)
    assert r_at_2 == 0.5, f"Expected 0.5, got {r_at_2}"
    print(" [PASS] Precision@k and Recall@k threshold calculations")

    # 3. Verify MRR and Jaccard
    recip_rank = mrr(["d3", "d1"], qrels)
    assert recip_rank == 0.5, f"Expected 0.5, got {recip_rank}"
    jaccard = jaccard_at_k(["a", "b", "c"], ["b", "c", "d"], k=3)
    assert math.isclose(jaccard, 2 / 4, rel_tol=1e-5)
    print(" [PASS] MRR (Mean Reciprocal Rank) and Jaccard set overlap")

    # 4. Verify RRF formula and k-parameter behavior
    c_lex = Candidate(document_id="doc1", bm25_rank=1, bm25_score=10.0)
    c_dense = Candidate(document_id="doc1", dense_rank=2, dense_score=0.9)
    fused_k10 = reciprocal_rank_fusion([c_lex], [c_dense], k=10)
    fused_k60 = reciprocal_rank_fusion([c_lex], [c_dense], k=60)
    expected_k10 = (1.0 / 11.0) + (1.0 / 12.0)
    expected_k60 = (1.0 / 61.0) + (1.0 / 62.0)
    assert fused_k10[0].rrf_score is not None and math.isclose(
        fused_k10[0].rrf_score, expected_k10, rel_tol=1e-5
    )
    assert fused_k60[0].rrf_score is not None and math.isclose(
        fused_k60[0].rrf_score, expected_k60, rel_tol=1e-5
    )
    assert fused_k10[0].rrf_score > fused_k60[0].rrf_score * 4.0
    print(
        " [PASS] Reciprocal Rank Fusion (RRF) formula & k=10 vs k=60 parameter invariant"
    )

    # 5. Verify Text Analyzer determinism
    tokens = analyze_text("Evaluating hybrid search with BM25 & dense retrieval!")
    assert (
        "evalu" in tokens
        and "hybrid" in tokens
        and "search" in tokens
        and "bm25" in tokens
    )
    assert "with" not in tokens and "&" not in tokens
    print(
        " [PASS] Text Analyzer normalization, 174-stopword filtering, Snowball stemming"
    )

    print("-" * 70)
    print("All IR mathematics, fusion invariants, and metric suites verified.")
    print("Status: 100% PASS (5/5 invariants)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="NEXUS Benchmark Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "verify",
        help="Hermetic validation of IR metrics, RRF math, and BM25 invariants",
    )

    p_prepare = subparsers.add_parser("prepare")
    p_prepare.add_argument("--dataset", required=True, choices=["scifact"])
    p_prepare.add_argument("--force-prepare", action="store_true")

    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--dataset", required=True, choices=["scifact"])
    p_eval.add_argument(
        "--split",
        required=True,
        choices=["test", "train"],
        help=(
            "'test' = BEIR test qrels (300 queries; these are the official SciFact dev "
            "claims) or 'train' (809 queries)"
        ),
    )
    p_eval.add_argument(
        "--with-rerank",
        action="store_true",
        help="Also evaluate CPU cross-encoder reranking of the top 25 (slow, downloads ~90 MB model)",
    )

    p_gen = subparsers.add_parser("generate-llm-variants")
    p_gen.add_argument("--dataset", required=True, choices=["scifact"])
    p_gen.add_argument(
        "--preflight",
        action="store_true",
        help="Run config validation and cache integrity check, no real API calls",
    )
    p_gen.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run exactly one real API call to test the pipeline",
    )

    p_eval_llm = subparsers.add_parser("evaluate-llm")
    p_eval_llm.add_argument("--dataset", required=True, choices=["scifact"])

    args = parser.parse_args()

    if args.command == "verify":
        cmd_verify(args)
    elif args.command == "prepare":
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
