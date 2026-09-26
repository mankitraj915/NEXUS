import copy
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from app.domains.search.query.classifier import classify_query
from app.domains.search.query.features import QueryFeatures
from app.domains.search.query.planner import generate_plan

OUTCOME_FILE = "scripts/router_training/action_outcome_matrix.jsonl"
EVAL_OUT = "docs/routing-evaluation.md"

CONFIG_NAMES = {
    "CONFIG_A": 0,
    "CONFIG_B": 1,
    "CONFIG_C": 2,
    "CONFIG_D": 3,
    "CONFIG_E": 4,
    "CONFIG_F": 5
}
INV_CONFIG_NAMES = {v: k for k, v in CONFIG_NAMES.items()}
CONFIG_DESC = {
    "CONFIG_A": "BM25=100, Dense=100",
    "CONFIG_B": "BM25=100, Dense=25",
    "CONFIG_C": "BM25=25, Dense=100",
    "CONFIG_D": "BM25=100, Dense=0",
    "CONFIG_E": "BM25=0, Dense=100",
    "CONFIG_F": "BM25=25, Dense=25"
}

def simulate_rule_based_router(features_dict):
    qf = QueryFeatures(**features_dict)
    classes = classify_query(qf)
    plan = generate_plan(qf, classes)
    b = plan.bm25_depth
    d = plan.dense_depth
    if b == 100 and d == 100: return "CONFIG_A"
    if b == 100 and d == 25: return "CONFIG_B"
    if b == 25 and d == 100: return "CONFIG_C"
    if b == 100 and d == 0: return "CONFIG_D"
    if b == 0 and d == 100: return "CONFIG_E"
    if b == 25 and d == 25: return "CONFIG_F"
    return "CONFIG_A" # default

def get_oracle_label(outcomes, epsilon):
    fixed = outcomes.get("CONFIG_A")
    if not fixed: return 0
    threshold = fixed["ndcg_10"] - epsilon
    best_action = 0
    min_lat = float('inf')
    for cfg_name, res in outcomes.items():
        if res["ndcg_10"] >= threshold and res["median_latency_ms"] < min_lat:
            min_lat = res["median_latency_ms"]
            best_action = CONFIG_NAMES[cfg_name]
    return best_action

def evaluate_predictions(y_pred, records, test_idx):
    res: dict[str, list[float]] = {
        "ndcg_5": [], "ndcg_10": [], "recall_10": [], "recall_100": [], "mrr": [],
        "lat_median": [], "lat_p75": [], "lat_p95_run": [], 
        "bm25_depth": [], "dense_depth": []
    }
    
    for i, idx in enumerate(test_idx):
        rec = records[idx]
        pred_cfg = INV_CONFIG_NAMES[y_pred[i]]
        outcome = rec["outcomes"][pred_cfg]
        
        res["ndcg_5"].append(outcome.get("ndcg_5", 0))
        res["ndcg_10"].append(outcome["ndcg_10"])
        res["recall_10"].append(outcome.get("recall_10", 0))
        res["recall_100"].append(outcome["recall_100"])
        res["mrr"].append(outcome["mrr"])
        res["lat_median"].append(outcome["median_latency_ms"])
        res["lat_p75"].append(outcome["p75_latency_ms"])
        res["lat_p95_run"].append(outcome.get("p95_latency_ms", outcome["p75_latency_ms"]))
        
        b = int(CONFIG_DESC[pred_cfg].split("BM25=")[1].split(",")[0])
        d = int(CONFIG_DESC[pred_cfg].split("Dense=")[1])
        res["bm25_depth"].append(b)
        res["dense_depth"].append(d)
        
    return {
        "ndcg_5": np.mean(res["ndcg_5"]),
        "ndcg_10": np.mean(res["ndcg_10"]),
        "recall_10": np.mean(res["recall_10"]),
        "recall_100": np.mean(res["recall_100"]),
        "mrr": np.mean(res["mrr"]),
        "lat_avg": np.mean(res["lat_median"]), # mean of medians
        "lat_p50": np.median(res["lat_median"]), # median of medians
        "lat_p95": np.percentile(res["lat_median"], 95),
        "lat_p99": np.percentile(res["lat_median"], 99),
        "avg_bm25": np.mean(res["bm25_depth"]),
        "avg_dense": np.mean(res["dense_depth"])
    }

def calculate_regrets(dt_pred, records):
    regrets_fixed = []
    regrets_oracle = []
    
    for i in range(len(dt_pred)):
        pred_cfg = INV_CONFIG_NAMES[dt_pred[i]]
        rec = records[i]
        
        l_ndcg = rec["outcomes"][pred_cfg]["ndcg_10"]
        f_ndcg = rec["outcomes"]["CONFIG_A"]["ndcg_10"]
        
        # Oracle A (Pure quality)
        o_ndcg = max(c["ndcg_10"] for c in rec["outcomes"].values())
        
        regrets_fixed.append(l_ndcg - f_ndcg)
        regrets_oracle.append(l_ndcg - o_ndcg)
        
    rf = np.array(regrets_fixed)
    return {
        "mean": np.mean(rf),
        "median": np.median(rf),
        "p90": np.percentile(rf, 10), # lowest values are worst regrets
        "p95": np.percentile(rf, 5),
        "p99": np.percentile(rf, 1),
        "worst": np.min(rf),
        "loss_005": np.mean(rf < -0.005),
        "loss_010": np.mean(rf < -0.010),
        "loss_050": np.mean(rf < -0.050),
        "loss_100": np.mean(rf < -0.100),
        "raw_fixed": regrets_fixed,
        "raw_oracle": regrets_oracle
    }

def main():
    if not os.path.exists(OUTCOME_FILE):
        print("Outcome matrix missing.")
        return
        
    records = []
    with open(OUTCOME_FILE, "r") as f:
        for line in f:
            records.append(json.loads(line))
            
    if not records:
        print("No records.")
        return
        
    feat_keys = sorted(records[0]["features"].keys())
    X = np.array([[r["features"].get(k, 0.0) for k in feat_keys] for r in records])
    groups = np.array([r["query_id"] for r in records])
    
    # Generate labels
    y_oracle_b_005 = np.array([get_oracle_label(r["outcomes"], 0.005) for r in records])
    
    y_oracle_a_list: list[int] = []
    for r in records:
        b_ndcg = -1.0
        b_lat = float('inf')
        b_act = 0
        for cfg, res_out in r["outcomes"].items():
            if res_out["ndcg_10"] > b_ndcg or (res_out["ndcg_10"] == b_ndcg and res_out["median_latency_ms"] < b_lat):
                b_ndcg = res_out["ndcg_10"]
                b_lat = res_out["median_latency_ms"]
                b_act = CONFIG_NAMES[cfg]
        y_oracle_a_list.append(b_act)
    y_oracle_a = np.array(y_oracle_a_list)
    
    y_rule_based = np.array([CONFIG_NAMES[simulate_rule_based_router(r["features"])] for r in records])
    y_fixed = np.zeros(len(records), dtype=int)
    
    gkf = GroupKFold(n_splits=5)
    
    # Train DT and RF targeting Oracle B (0.005)
    dt_pred = np.zeros(len(records), dtype=int)
    rf_pred = np.zeros(len(records), dtype=int)
    
    fold_metrics = []
    
    dt_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=5, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_oracle_b_005, groups)):
        dt = copy.deepcopy(dt_model)
        rf = copy.deepcopy(rf_model)
        
        dt.fit(X[train_idx], y_oracle_b_005[train_idx])
        rf.fit(X[train_idx], y_oracle_b_005[train_idx])
        
        p_dt = dt.predict(X[test_idx])
        dt_pred[test_idx] = p_dt
        rf_pred[test_idx] = rf.predict(X[test_idx])
        
        # fold stability metric for DT
        fm = evaluate_predictions(p_dt, records, test_idx)
        reg = calculate_regrets(p_dt, [records[i] for i in test_idx])
        fm["mean_regret"] = reg["mean"]
        fold_metrics.append(fm)
        
    # Global metrics
    res_fixed = evaluate_predictions(y_fixed, records, range(len(records)))
    res_rule = evaluate_predictions(y_rule_based, records, range(len(records)))
    res_dt = evaluate_predictions(dt_pred, records, range(len(records)))
    res_rf = evaluate_predictions(rf_pred, records, range(len(records)))
    res_ora = evaluate_predictions(y_oracle_a, records, range(len(records)))
    res_orb = evaluate_predictions(y_oracle_b_005, records, range(len(records)))
    
    # Regrets
    dt_regrets = calculate_regrets(dt_pred, records)
    
    # Worst case 20
    worst_idx = np.argsort(dt_regrets["raw_fixed"])[:20]
    worst_cases_md = ""
    for i in worst_idx:
        rec = records[i]
        reg = dt_regrets["raw_fixed"][i]
        if reg >= 0: continue
        
        q = rec["query_text"]
        pred_cfg = INV_CONFIG_NAMES[dt_pred[i]]
        orac_cfg = INV_CONFIG_NAMES[y_oracle_b_005[i]]
        f_ndcg = rec["outcomes"]["CONFIG_A"]["ndcg_10"]
        p_ndcg = rec["outcomes"][pred_cfg]["ndcg_10"]
        f_lat = rec["outcomes"]["CONFIG_A"]["median_latency_ms"]
        p_lat = rec["outcomes"][pred_cfg]["median_latency_ms"]
        b = CONFIG_DESC[pred_cfg]
        
        worst_cases_md += f"**Query:** `{q}`\n"
        worst_cases_md += f"- **Delta NDCG:** {reg:.4f} ({f_ndcg:.4f} -> {p_ndcg:.4f})\n"
        worst_cases_md += f"- **Latency:** {f_lat:.2f}ms -> {p_lat:.2f}ms\n"
        worst_cases_md += f"- **Selected:** {pred_cfg} ({b}) | **Oracle:** {orac_cfg}\n\n"
        
    # Feature Importance
    dt_model.fit(X, y_oracle_b_005)
    importances = dt_model.feature_importances_
    feat_imp = sorted(zip(feat_keys, importances), key=lambda x: x[1], reverse=True)
    fi_md = "\n".join([f"- **{k}**: {v:.4f}" for k, v in feat_imp if v > 0])
    
    # Probe ablation
    probe_lat = np.mean([r["probe"].get("median_latency_ms", 0) for r in records])
    
    md = f"""# NEXUS Phase 7.5 Validation Report

## 1. Executive Summary
This report details the full GroupKFold validation of the Learned Routing Models on the SciFact benchmark. 
The Decision Tree achieved significant latency savings but we must inspect the worst-case degradation.

## 2. Dataset
- **Queries Evaluated**: {len(records)}
- **Configurations**: 6 (A through F)
- **Leakage Safe**: GroupKFold by query_id (5 Folds)

## 3. Overall Performance

| Policy | NDCG@5 | NDCG@10 | Recall@100 | MRR | Latency (Avg) | Latency (P50) | Latency (P95) | Avg BM25 | Avg Dense |
|---|---|---|---|---|---|---|---|---|---|
| Fixed Baseline | {res_fixed['ndcg_5']:.4f} | {res_fixed['ndcg_10']:.4f} | {res_fixed['recall_100']:.4f} | {res_fixed['mrr']:.4f} | {res_fixed['lat_avg']:.2f}ms | {res_fixed['lat_p50']:.2f}ms | {res_fixed['lat_p95']:.2f}ms | {res_fixed['avg_bm25']:.1f} | {res_fixed['avg_dense']:.1f} |
| Rule-Based | {res_rule['ndcg_5']:.4f} | {res_rule['ndcg_10']:.4f} | {res_rule['recall_100']:.4f} | {res_rule['mrr']:.4f} | {res_rule['lat_avg']:.2f}ms | {res_rule['lat_p50']:.2f}ms | {res_rule['lat_p95']:.2f}ms | {res_rule['avg_bm25']:.1f} | {res_rule['avg_dense']:.1f} |
| Decision Tree | {res_dt['ndcg_5']:.4f} | {res_dt['ndcg_10']:.4f} | {res_dt['recall_100']:.4f} | {res_dt['mrr']:.4f} | {res_dt['lat_avg']:.2f}ms | {res_dt['lat_p50']:.2f}ms | {res_dt['lat_p95']:.2f}ms | {res_dt['avg_bm25']:.1f} | {res_dt['avg_dense']:.1f} |
| Random Forest | {res_rf['ndcg_5']:.4f} | {res_rf['ndcg_10']:.4f} | {res_rf['recall_100']:.4f} | {res_rf['mrr']:.4f} | {res_rf['lat_avg']:.2f}ms | {res_rf['lat_p50']:.2f}ms | {res_rf['lat_p95']:.2f}ms | {res_rf['avg_bm25']:.1f} | {res_rf['avg_dense']:.1f} |
| Oracle A (Max Quality) | {res_ora['ndcg_5']:.4f} | {res_ora['ndcg_10']:.4f} | {res_ora['recall_100']:.4f} | {res_ora['mrr']:.4f} | {res_ora['lat_avg']:.2f}ms | {res_ora['lat_p50']:.2f}ms | {res_ora['lat_p95']:.2f}ms | {res_ora['avg_bm25']:.1f} | {res_ora['avg_dense']:.1f} |
| Oracle B (Constrained) | {res_orb['ndcg_5']:.4f} | {res_orb['ndcg_10']:.4f} | {res_orb['recall_100']:.4f} | {res_orb['mrr']:.4f} | {res_orb['lat_avg']:.2f}ms | {res_orb['lat_p50']:.2f}ms | {res_orb['lat_p95']:.2f}ms | {res_orb['avg_bm25']:.1f} | {res_orb['avg_dense']:.1f} |

## 4. Regret Analysis (Decision Tree vs Fixed)
- **Mean Regret**: {dt_regrets['mean']:.4f}
- **Median Regret**: {dt_regrets['median']:.4f}
- **P90 Worst-case**: {dt_regrets['p90']:.4f}
- **P95 Worst-case**: {dt_regrets['p95']:.4f}
- **P99 Worst-case**: {dt_regrets['p99']:.4f}
- **Absolute Worst**: {dt_regrets['worst']:.4f}

**Loss Breakdown (Percentage of queries degrading by X):**
- Loss > 0.005: {dt_regrets['loss_005']*100:.1f}%
- Loss > 0.010: {dt_regrets['loss_010']*100:.1f}%
- Loss > 0.050: {dt_regrets['loss_050']*100:.1f}%
- Loss > 0.100: {dt_regrets['loss_100']*100:.1f}%

## 5. Cross-Fold Stability (Decision Tree)
| Fold | NDCG@10 | Latency (Avg) | Mean Regret |
|---|---|---|---|
"""
    for i, fm in enumerate(fold_metrics):
        md += f"| {i+1} | {fm['ndcg_10']:.4f} | {fm['lat_avg']:.2f}ms | {fm['mean_regret']:.4f} |\n"
        
    ndcgs = [fm['ndcg_10'] for fm in fold_metrics]
    md += f"\n**NDCG Stats**: Mean={np.mean(ndcgs):.4f}, Std={np.std(ndcgs):.4f}, Min={np.min(ndcgs):.4f}, Max={np.max(ndcgs):.4f}\n"

    md += f"""
## 6. Worst-Case Queries
{worst_cases_md}

## 7. Probe Ablation
- Average Probe Latency Overhead: {probe_lat:.2f}ms
(If probe overhead > latency savings on affected queries, it should be disabled.)

## 8. Feature Importance
{fi_md}

## 9. Final Recommendation
(To be updated based on results)
"""
    with open(EVAL_OUT, "w", encoding='utf-8') as f:
        f.write(md)
        
    print(f"Evaluation complete. Wrote {EVAL_OUT}")

if __name__ == "__main__":
    main()
