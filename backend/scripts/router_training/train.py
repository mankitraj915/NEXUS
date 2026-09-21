import json
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

OUTCOME_FILE = "scripts/router_training/action_outcome_matrix.jsonl"
MODEL_OUT = "scripts/router_training/router.pkl"

def get_action_index(cfg_name):
    # Matches model.py ACTIONS
    mapping = {
        "CONFIG_A": 0,
        "CONFIG_B": 1,
        "CONFIG_C": 2,
        "CONFIG_D": 3,
        "CONFIG_E": 4,
        "CONFIG_F": 5
    }
    return mapping[cfg_name]

def label_oracle_b(outcomes, epsilon=0.005):
    """
    Quality-Constrained Latency Oracle
    Find all configs where NDCG@10 >= Fixed_NDCG@10 - epsilon.
    Among those, return the one with minimum median_latency_ms.
    """
    fixed = outcomes.get("CONFIG_A")
    if not fixed:
        return 0 # Fallback
        
    baseline_ndcg = fixed["ndcg_10"]
    threshold = baseline_ndcg - epsilon
    
    best_action = 0
    min_latency = float('inf')
    
    for cfg_name, results in outcomes.items():
        if results["ndcg_10"] >= threshold:
            if results["median_latency_ms"] < min_latency:
                min_latency = results["median_latency_ms"]
                best_action = get_action_index(cfg_name)
                
    return best_action

def evaluate_predictions(y_true, y_pred, y_fixed, all_outcomes, test_idx, data_records):
    """
    Evaluates predictions based on actual outcomes, not just classification accuracy.
    """
    # This is a bit tricky, we need to map predictions back to the actual ndcg and latency
    avg_ndcg = []
    avg_recall = []
    avg_lat = []
    
    fixed_ndcg = []
    fixed_lat = []
    
    oracle_ndcg = []
    oracle_lat = []
    
    cfg_names = {v: k for k, v in {"CONFIG_A": 0, "CONFIG_B": 1, "CONFIG_C": 2, 
                                   "CONFIG_D": 3, "CONFIG_E": 4, "CONFIG_F": 5}.items()}
    
    for i, idx in enumerate(test_idx):
        record = data_records[idx]
        outcomes = record["outcomes"]
        
        # Learned Router
        pred_action = y_pred[i]
        pred_cfg = cfg_names[pred_action]
        res = outcomes[pred_cfg]
        avg_ndcg.append(res["ndcg_10"])
        avg_recall.append(res["recall_100"])
        avg_lat.append(res["median_latency_ms"])
        
        # Fixed
        fixed_res = outcomes["CONFIG_A"]
        fixed_ndcg.append(fixed_res["ndcg_10"])
        fixed_lat.append(fixed_res["median_latency_ms"])
        
        # Oracle B (y_true holds the oracle label)
        oracle_cfg = cfg_names[y_true[i]]
        or_res = outcomes[oracle_cfg]
        oracle_ndcg.append(or_res["ndcg_10"])
        oracle_lat.append(or_res["median_latency_ms"])
        
    return {
        "ndcg": np.mean(avg_ndcg),
        "recall": np.mean(avg_recall),
        "lat_mean": np.mean(avg_lat),
        "lat_p95": np.percentile(avg_lat, 95) if avg_lat else 0,
        "fixed_ndcg": np.mean(fixed_ndcg),
        "fixed_lat_mean": np.mean(fixed_lat),
        "oracle_ndcg": np.mean(oracle_ndcg),
        "oracle_lat_mean": np.mean(oracle_lat),
        "accuracy": accuracy_score(y_true, y_pred)
    }

def train_and_evaluate():
    if not os.path.exists(OUTCOME_FILE):
        print("Outcome matrix not found. Run dataset.py first.")
        return
        
    records = []
    with open(OUTCOME_FILE, "r") as f:
        for line in f:
            records.append(json.loads(line))
            
    if not records:
        print("No records found.")
        return
        
    feature_keys = sorted(list(records[0]["features"].keys()))
    X = []
    y_oracle_b = [] # Quality-constrained latency
    y_oracle_a = [] # Pure quality
    
    epsilon = 0.005 # configurable tolerance
    
    for r in records:
        vec = [r["features"].get(k, 0.0) for k in feature_keys]
        X.append(vec)
        y_oracle_b.append(label_oracle_b(r["outcomes"], epsilon=epsilon))
        
        # Oracle A: pure max ndcg, tie-break lowest latency
        best_ndcg = -1
        best_lat = float('inf')
        best_a = 0
        for cfg_name, res in r["outcomes"].items():
            if res["ndcg_10"] > best_ndcg or (res["ndcg_10"] == best_ndcg and res["median_latency_ms"] < best_lat):
                best_ndcg = res["ndcg_10"]
                best_lat = res["median_latency_ms"]
                best_a = get_action_index(cfg_name)
        y_oracle_a.append(best_a)
        
    X = np.array(X)
    y = np.array(y_oracle_b)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=5, random_state=42)
    }
    
    results_summary = {}
    
    # Store predictions for regret analysis
    best_model_name = "Decision Tree"
    
    for name, model_cls in models.items():
        all_y_pred = np.zeros_like(y)
        for train_idx, test_idx in kf.split(X):
            model_cls.fit(X[train_idx], y[train_idx])
            all_y_pred[test_idx] = model_cls.predict(X[test_idx])
            
        res = evaluate_predictions(y, all_y_pred, np.zeros_like(y), records, range(len(X)), records)
        results_summary[name] = res
        print(f"{name}: NDCG={res['ndcg']:.4f} Latency={res['lat_mean']:.2f}ms")
        
    # Full dataset evaluation for Fixed, Oracle
    fixed_res = evaluate_predictions(y, np.zeros_like(y), np.zeros_like(y), records, range(len(X)), records)
    oracle_b_res = evaluate_predictions(y, y, np.zeros_like(y), records, range(len(X)), records)
    oracle_a_res = evaluate_predictions(np.array(y_oracle_a), np.array(y_oracle_a), np.zeros_like(y), records, range(len(X)), records)
    
    # Let's perform Regret Analysis for Decision Tree vs Fixed Baseline
    dt_pred = np.zeros_like(y)
    for train_idx, test_idx in kf.split(X):
        m = models["Decision Tree"]
        m.fit(X[train_idx], y[train_idx])
        dt_pred[test_idx] = m.predict(X[test_idx])
        
    regrets = []
    cfg_names = {v: k for k, v in {"CONFIG_A": 0, "CONFIG_B": 1, "CONFIG_C": 2, "CONFIG_D": 3, "CONFIG_E": 4, "CONFIG_F": 5}.items()}
    for i in range(len(dt_pred)):
        pred_cfg = cfg_names[dt_pred[i]]
        fixed_ndcg = records[i]["outcomes"]["CONFIG_A"]["ndcg_10"]
        pred_ndcg = records[i]["outcomes"][pred_cfg]["ndcg_10"]
        regrets.append(pred_ndcg - fixed_ndcg)
        
    mean_regret = np.mean(regrets)
    median_regret = np.median(regrets)
    p95_regret = np.percentile(regrets, 5) # 5th percentile is the P95 worst-case loss
    worst_regret = np.min(regrets)
    
    print("\n--- REGRET ANALYSIS (Decision Tree vs Fixed) ---")
    print(f"Mean Regret: {mean_regret:.4f}")
    print(f"Median Regret: {median_regret:.4f}")
    print(f"P95 Worst-case Loss: {p95_regret:.4f}")
    print(f"Worst-case Regret: {worst_regret:.4f}")
    
    report_content = f"""# NEXUS Phase 7 Learned Routing Evaluation

## Executive Summary
This report evaluates the performance of a learned routing model designed to optimize retrieval depth using a constrained action space. The learned models (Decision Tree & Random Forest) were trained to predict a Quality-Constrained Latency Oracle label.

## Action Space
- **CONFIG_A**: BM25=100, Dense=100
- **CONFIG_B**: BM25=100, Dense=25
- **CONFIG_C**: BM25=25, Dense=100
- **CONFIG_D**: BM25=100, Dense=0
- **CONFIG_E**: BM25=0, Dense=100
- **CONFIG_F**: BM25=25, Dense=25

## Overall Pareto Frontier

| Policy | NDCG@10 | Recall@100 | Latency Avg | P95 |
|---|---|---|---|---|
| Fixed Baseline | {fixed_res['fixed_ndcg']:.4f} | - | {fixed_res['fixed_lat_mean']:.2f}ms | - |
| Rule-Based | - | - | - | - |
| Decision Tree | {results_summary['Decision Tree']['ndcg']:.4f} | {results_summary['Decision Tree']['recall']:.4f} | {results_summary['Decision Tree']['lat_mean']:.2f}ms | {results_summary['Decision Tree']['lat_p95']:.2f}ms |
| Random Forest | {results_summary['Random Forest']['ndcg']:.4f} | {results_summary['Random Forest']['recall']:.4f} | {results_summary['Random Forest']['lat_mean']:.2f}ms | {results_summary['Random Forest']['lat_p95']:.2f}ms |
| Quality Oracle | {oracle_a_res['ndcg']:.4f} | {oracle_a_res['recall']:.4f} | {oracle_a_res['lat_mean']:.2f}ms | {oracle_a_res['lat_p95']:.2f}ms |
| Constrained Oracle | {oracle_b_res['ndcg']:.4f} | {oracle_b_res['recall']:.4f} | {oracle_b_res['lat_mean']:.2f}ms | {oracle_b_res['lat_p95']:.2f}ms |

## Regret Analysis (Decision Tree vs Fixed)
- Mean Regret: {mean_regret:.4f}
- Median Regret: {median_regret:.4f}
- P95 Worst-case Loss: {p95_regret:.4f}
- Worst-case Regret: {worst_regret:.4f}
"""
    with open("docs/routing-evaluation.md", "w") as f:
        f.write(report_content)

    # Train final model on full dataset
    final_model = models["Decision Tree"]
    final_model.fit(X, y)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump((final_model, feature_keys), f)
        
    print(f"\nModel saved to {MODEL_OUT}")

if __name__ == "__main__":
    train_and_evaluate()
