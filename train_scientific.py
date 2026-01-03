"""
================================================================================
SCIENTIFIC EVALUATION v2.1
================================================================================
- Respektiert neue Ordnerstruktur (data/, results/)
- Ergebnisse werden in results/ gespeichert
- CSV wird in data/ oder NEO4J_IMPORT_DIR gesucht
================================================================================
"""

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import json
from sklearn.metrics import average_precision_score, roc_auc_score

# Config importieren
try:
    import config
    DATA_DIR = config.DATA_DIR
    RESULTS_DIR = config.RESULTS_DIR
    NEO4J_IMPORT_DIR = config.NEO4J_IMPORT_DIR
except ImportError:
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    NEO4J_IMPORT_DIR = "."

# Bridging Features (optional)
try:
    from bridging_features import add_all_bridging_features, BRIDGING_FEATURE_NAMES
    BRIDGING_AVAILABLE = True
except ImportError:
    BRIDGING_AVAILABLE = False
    BRIDGING_FEATURE_NAMES = []

# ==========================================
# KONFIGURATION
# ==========================================
CSV_FILENAME = os.environ.get("CSV_FILENAME", config.CSV_FILENAME if 'config' in dir() else "")
GRAPH_FEAT_FILE = "graph_features_comparison.csv"

# Results File - in results/ speichern
results_filename = os.environ.get("RESULTS_FILENAME", "results.json")
RESULTS_FILE = os.path.join(RESULTS_DIR, os.path.basename(results_filename))

OOS_PERCENTAGE = 0.05
SEEDS = [42, 101, 2024, 7, 99]


# ==========================================
# HELPER: Datei finden
# ==========================================
def find_file(filename: str) -> str:
    """Sucht Datei in data/, results/, NEO4J_IMPORT_DIR, current dir."""
    candidates = [
        os.path.join(DATA_DIR, filename),
        os.path.join(RESULTS_DIR, filename),
        os.path.join(NEO4J_IMPORT_DIR, filename),
        filename,
        os.path.basename(filename)
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    return filename  # Return original if not found


# ==========================================
# 1. DATEN LADEN & MERGEN
# ==========================================
print("🔬 STARTING SCIENTIFIC EVALUATION...")
print(f"   📂 Results will be saved to: {RESULTS_FILE}")

# Graph Features laden
graph_feat_path = find_file(GRAPH_FEAT_FILE)
if not os.path.exists(graph_feat_path):
    print(f"❌ ERROR: '{GRAPH_FEAT_FILE}' not found.")
    print(f"   Searched: {graph_feat_path}")
    sys.exit(1)

print(f"   → Loading graph features from: {graph_feat_path}")
df_graph = pd.read_csv(graph_feat_path)

# Transaction Data laden
trans_path = find_file(CSV_FILENAME)
if not os.path.exists(trans_path):
    print(f"❌ ERROR: Transaction file not found: {CSV_FILENAME}")
    print(f"   Searched: {trans_path}")
    sys.exit(1)

print(f"   → Loading transactions from: {trans_path}")
df_trans = pd.read_csv(trans_path, low_memory=False)

# Merge
df_trans['GPID'] = df_trans['GPID'].astype(str)
df_graph['customer_id'] = df_graph['customer_id'].astype(str)
df = df_trans.merge(df_graph, left_on="GPID", right_on="customer_id", how="left", suffixes=('', '_graph'))
df.drop(columns=['customer_id'], inplace=True, errors='ignore')

# Cleanup duplicate columns
for col in ['risk_score']:
    if f'{col}_graph' in df.columns:
        df.drop(columns=[f'{col}_graph'], inplace=True)

df = df.dropna(subset=['Fraud_incl_wo'])
df['Fraud_incl_wo'] = df['Fraud_incl_wo'].astype(int)

print(f"   → Merged dataset: {len(df):,} rows")

# Temporal Sort
if 'PRUEFUNG_INTERN_DATUM' in df.columns:
    df['_timestamp'] = pd.to_datetime(df['PRUEFUNG_INTERN_DATUM'], errors='coerce')
    df = df.sort_values('_timestamp').reset_index(drop=True)
    df.drop(columns=['_timestamp'], inplace=True)


# ==========================================
# 2. FEATURE DEFINITION
# ==========================================
exclude_cols = [
    'GPID', 'Fraud_incl_wo', 'RMS_PRUEF_ID', 'EMAIL', 
    'TMX_DIGITAL_ID', 'TMX_DEV_IPADDRESS', 'SESSION_ID', 'RMS_PRUEF_ID_MASTER', 
    'fold', 'FIRST_NAME', 'LAST_NAME', 'RA_STRASSE', 'LA_STRASSE', 
    'RA_ORT', 'LA_ORT', 'RA_PLZ', 'LA_PLZ', 'ADDRESS_HASH', 
    'PRUEFUNG_INTERN_DATUM'
]
all_cols = df.columns.tolist()

# Feature Sets
feat_topo = [c for c in all_cols if "_full" in c and any(x in c for x in ["degree", "wccId", "triangles", "lcc"])]
feat_emb = [c for c in all_cols if any(x in c for x in ["fastrp_full", "sage_full", "n2v_full"])]
graph_cols = feat_topo + feat_emb
feat_base = [c for c in all_cols if c not in exclude_cols and c not in graph_cols and "_trunc" not in c and "_graph" not in c]

print(f"   → Base features: {len(feat_base)}")
print(f"   → Topology features: {len(feat_topo)}")
print(f"   → Embedding features: {len(feat_emb)}")

# Bridging Features
feat_bridge = []
if BRIDGING_AVAILABLE and 'degree_full' in df.columns and 'wccId_full' in df.columns:
    try:
        df = add_all_bridging_features(df, verbose=False)
        feat_bridge = [c for c in BRIDGING_FEATURE_NAMES if c in df.columns]
        print(f"   → Bridging features: {len(feat_bridge)}")
    except Exception as e:
        print(f"   ⚠️ Bridging calc failed: {e}")


# ==========================================
# 3. PREPROCESSING & SPLIT
# ==========================================
# Fill NA
for col in graph_cols + feat_base + feat_bridge:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Temporal Split
split_idx = int(len(df) * (1 - OOS_PERCENTAGE))
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()
y_train = df_train['Fraud_incl_wo']
y_test = df_test['Fraud_incl_wo']

print(f"   → Train: {len(df_train):,} | Test: {len(df_test):,}")
print(f"   → Fraud rate: Train={y_train.mean():.2%} | Test={y_test.mean():.2%}")

# Encode categorical
for c in feat_base:
    if c in df_train.columns and df_train[c].dtype == 'object':
        mapping = {val: idx for idx, val in enumerate(df_train[c].astype(str).unique())}
        df_train[c] = df_train[c].astype(str).map(mapping).fillna(-1).astype(int)
        df_test[c] = df_test[c].astype(str).map(mapping).fillna(-1).astype(int)

pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)


# ==========================================
# 4. TRAINING
# ==========================================
experiments = {"Baseline": feat_base}
if feat_emb: 
    experiments["Embeddings"] = feat_base + feat_emb
if feat_topo: 
    experiments["Topology"] = feat_base + feat_topo
if feat_topo and feat_emb: 
    experiments["Hybrid"] = feat_base + feat_topo + feat_emb
if feat_bridge: 
    experiments["Bridge"] = feat_base + feat_bridge
if feat_topo and feat_emb and feat_bridge: 
    experiments["Full"] = feat_base + feat_topo + feat_emb + feat_bridge

# Fallback: Wenn keine Graph-Features, Full = Baseline
if "Full" not in experiments:
    experiments["Full"] = feat_base

results_export = {}
best_model = None
best_features = None

print(f"\n{'Model':<20} | {'PR-AUC':<20} | {'ROC-AUC':<20}")
print("-" * 65)

for name, feats in experiments.items():
    feats = [f for f in feats if f in df_train.columns]
    if not feats:
        continue
    
    pr_scores, roc_scores = [], []
    
    for seed in SEEDS:
        clf = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.05,
            scale_pos_weight=pos_weight, 
            eval_metric='aucpr',
            n_jobs=-1, 
            random_state=seed, 
            tree_method='hist'
        )
        clf.fit(df_train[feats], y_train, verbose=False)
        probs = clf.predict_proba(df_test[feats])[:, 1]
        pr_scores.append(average_precision_score(y_test, probs))
        roc_scores.append(roc_auc_score(y_test, probs))
        
        if name == "Full" and seed == SEEDS[0]:
            best_model = clf
            best_features = feats

    pr_m, pr_s = np.mean(pr_scores), np.std(pr_scores)
    roc_m, roc_s = np.mean(roc_scores), np.std(roc_scores)
    results_export[name] = {
        "pr_mean": float(pr_m), 
        "pr_std": float(pr_s), 
        "roc_mean": float(roc_m), 
        "roc_std": float(roc_s)
    }
    
    print(f"{name:<20} | {pr_m:.4f} ± {pr_s:.4f}  | {roc_m:.4f} ± {roc_s:.4f}")


# ==========================================
# 5. EXPORT RESULTS
# ==========================================
# Stelle sicher, dass results/ existiert
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(RESULTS_FILE, 'w') as f:
    json.dump(results_export, f, indent=4)
print(f"\n✅ Results saved to {RESULTS_FILE}")


# ==========================================
# 6. SHAP (Optional)
# ==========================================
current_db = os.environ.get("NEO4J_DB", "neo4j")
if "benchmark" not in current_db and best_model and best_features:
    print("\n🎨 Generating SHAP...")
    try:
        import shap
        test_sample = df_test[best_features].sample(n=min(1000, len(df_test)), random_state=42)
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(test_sample)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, test_sample, plot_type="bar", max_display=15, show=False)
        
        # SHAP Plot in results/ speichern
        shap_path = os.path.join(RESULTS_DIR, "shap_bar_final.png")
        plt.savefig(shap_path, bbox_inches='tight')
        print(f"   ✅ SHAP plot saved to {shap_path}")
    except Exception as e:
        print(f"   ⚠️ SHAP skipped: {e}")


# ==========================================
# 7. SUMMARY
# ==========================================
print("\n" + "="*65)
print("SUMMARY")
print("="*65)

if "Baseline" in results_export and "Full" in results_export:
    baseline = results_export["Baseline"]["pr_mean"]
    full = results_export["Full"]["pr_mean"]
    lift = (full - baseline) * 100
    print(f"Baseline PR-AUC: {baseline:.4f}")
    print(f"Full PR-AUC:     {full:.4f}")
    print(f"Lift:            {lift:+.2f}%")
print("="*65)
