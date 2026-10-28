class ExperimentTracker:
    def __init__(self) -> None:
        self.metrics: dict[str, list[float]] = {
            "ndcg@5": [],
            "ndcg@10": [],
            "recall@5": [],
            "recall@10": [],
            "recall@20": [],
            "recall@100": [],
            "mrr": [],
            "latency": [],
        }
        self.query_ids: list[str] = []

    def add_latency(self, latency_ms: float) -> None:
        self.metrics["latency"].append(latency_ms)

    def add_metrics(
        self, ext_ids: list[str], qrels: dict[str, int], query_id: str = ""
    ) -> None:
        from scripts.benchmark.metrics import mrr, ndcg_at_k, recall_at_k

        self.query_ids.append(query_id)
        self.metrics["ndcg@5"].append(ndcg_at_k(ext_ids, qrels, k=5))
        self.metrics["ndcg@10"].append(ndcg_at_k(ext_ids, qrels, k=10))
        self.metrics["recall@5"].append(recall_at_k(ext_ids, qrels, k=5, threshold=1))
        self.metrics["recall@10"].append(recall_at_k(ext_ids, qrels, k=10, threshold=1))
        self.metrics["recall@20"].append(recall_at_k(ext_ids, qrels, k=20, threshold=1))
        self.metrics["recall@100"].append(
            recall_at_k(ext_ids, qrels, k=100, threshold=1)
        )
        self.metrics["mrr"].append(mrr(ext_ids, qrels))

    def per_query(self) -> dict[str, list]:
        """Per-query scores in evaluation order, for paired significance tests."""
        return {
            "query_ids": self.query_ids,
            **{
                m: [round(v, 6) for v in self.metrics[m]]
                for m in ("ndcg@10", "recall@100", "mrr")
            },
        }

    def get_summary(self) -> dict[str, float]:
        def avg(l: list[float]) -> float:
            return sum(l) / len(l) if l else 0.0

        def p(l: list[float], p_val: float) -> float:
            if not l:
                return 0.0
            s = sorted(l)
            idx = int(len(s) * p_val)
            return s[min(idx, len(s) - 1)]

        return {
            "ndcg@5": avg(self.metrics["ndcg@5"]),
            "ndcg@10": avg(self.metrics["ndcg@10"]),
            "recall@5": avg(self.metrics["recall@5"]),
            "recall@10": avg(self.metrics["recall@10"]),
            "recall@20": avg(self.metrics["recall@20"]),
            "recall@100": avg(self.metrics["recall@100"]),
            "mrr": avg(self.metrics["mrr"]),
            "latency_p50": p(self.metrics["latency"], 0.50),
            "latency_p95": p(self.metrics["latency"], 0.95),
            "latency_p99": p(self.metrics["latency"], 0.99),
            "latency_avg": avg(self.metrics["latency"]),
        }
