import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, PrecisionRecallDisplay

# ==========================================
# 1. SETUP & LOADING
# ==========================================
print("Loading data...")

# Check if feature file exists (output from Step 3)
if not os.path.exists("graph_features_final.csv"):
    print("ERROR: 'graph_features_final.csv' not found!")
    print("Please run 'graph_feature_gen.py' first.")
    sys.exit(1)

# Load Datasets
if os.path.exists("transactions.csv"):
    trans_file = "transactions.csv"
elif os.path.exists("synthetic_fraud_graph_data.csv"):
    trans_file = "synthetic_fraud_graph_data.csv"
else:
    print("ERROR: Could not find transaction data (transactions.csv)!")
    sys.exit(1)

# Load with type inference
df_trans = pd.read_csv(trans_file, low_memory=False)
df_graph = pd.read_csv("graph_features_final.csv")

print(f"Transactions: {df_trans.shape}")
print(f"Graph Features: {df_graph.shape}")

# --- FIX: TYPE MISMATCH ---
# Ensure join keys are strictly strings to prevent Object vs Float errors
df_trans['GPID'] = df_trans['GPID'].astype(str)
df_graph['customer_id'] = df_graph['customer_id'].astype(str)

# Merge
print("Merging datasets...")
# We left join to keep all transactions (even cold start/isolated ones)
df = df_trans.merge(df_graph, left_on="GPID", right_on="customer_id", how="left")

# Handle Cold Start (Missing Graph Features) with distinct value (-1)
df.fillna(-1, inplace=True)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("Preparing features...")

# Columns to exclude from training (IDs, PII, Leakage)
exclude_cols = [
    'GPID', 'customer_id', 'Fraud_incl_wo', 'RMS_PRUEF_ID',
    'EMAIL', 'TMX_DIGITAL_ID', 'TMX_DEV_IPADDRESS', 'SESSION_ID',
    'RMS_PRUEF_ID_MASTER', 'fold', 'FIRST_NAME', 'LAST_NAME',
    'RA_STRASSE', 'LA_STRASSE', 'RA_ORT', 'LA_ORT', 'RA_PLZ', 'LA_PLZ',
    'ADDRESS_HASH'  # Exclude if present
]

# Identify Categorical Columns (Strings) for Label Encoding
cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in exclude_cols]
print(f"Encoding categorical columns: {cat_cols}")

for c in cat_cols:
    df[c] = df[c].astype('category').cat.codes

# --- DEFINE FEATURE TIERS ---

# 1. Baseline: All numeric/encoded columns EXCEPT graph features
all_graph_cols = ["community_size", "degree_centrality"] + [c for c in df.columns if "sage_" in c]
feats_baseline = [c for c in df.columns if c not in exclude_cols and c not in all_graph_cols]

# 2. Tier 1: Baseline + Explicit Structure (Scalars)
feats_tier1 = feats_baseline + ["community_size", "degree_centrality"]

# 3. Tier 2: Baseline + Structure + Embeddings (Latent)
feats_tier2 = feats_tier1 + [c for c in df.columns if "sage_" in c]

print(f"\nFeature Counts:")
print(f"  Baseline: {len(feats_baseline)}")
print(f"  Tier 1:   {len(feats_tier1)} (+Structure)")
print(f"  Tier 2:   {len(feats_tier2)} (+Embeddings)")

# ==========================================
# 3. SPLIT & TRAIN
# ==========================================
# Stratified Split to maintain fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    df,
    df['Fraud_incl_wo'],
    test_size=0.2,
    stratify=df['Fraud_incl_wo'],
    random_state=42
)

results = {}


def run_model(name, feature_cols):
    print(f"\n--- Training {name} ---")

    # Standard Fraud Detection Hyperparameters
    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,  # Crucial for Imbalanced Data
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train[feature_cols], y_train)
    probs = clf.predict_proba(X_test[feature_cols])[:, 1]

    # Metrics
    auc_pr = average_precision_score(y_test, probs)
    auc_roc = roc_auc_score(y_test, probs)

    results[name] = {"probs": probs, "auc_pr": auc_pr}

    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  AUC-PR:  {auc_pr:.4f} <--- Key Metric")

    return clf


# Execute the Methodology
model_base = run_model("Baseline (XGB)", feats_baseline)
model_tier1 = run_model("Tier 1 (+Topology)", feats_tier1)
model_tier2 = run_model("Tier 2 (+Embeddings)", feats_tier2)

# ==========================================
# 4. REPORTING & VISUALIZATION
# ==========================================
print("\n--- Final Analysis ---")
base_score = results["Baseline (XGB)"]["auc_pr"]
tier1_score = results["Tier 1 (+Topology)"]["auc_pr"]
tier2_score = results["Tier 2 (+Embeddings)"]["auc_pr"]

lift_t1 = (tier1_score - base_score) / base_score * 100
lift_t2 = (tier2_score - base_score) / base_score * 100

print(f"Lift Tier 1 (Structure): {lift_t1:+.2f}%")
print(f"Lift Tier 2 (Full Graph): {lift_t2:+.2f}%")

print("\nGenerating interactive plots...")

# PLOT 1: Precision-Recall Curve Comparison
plt.figure(figsize=(10, 6))
for name, data in results.items():
    PrecisionRecallDisplay.from_predictions(
        y_test,
        data['probs'],
        name=f"{name} (AP={data['auc_pr']:.3f})",
        ax=plt.gca()
    )

plt.title(f"Methodology Proof: Graph Lift (+{lift_t2:.1f}%)")
plt.grid(True, alpha=0.3)
plt.show()

# PLOT 2: Feature Importance (Tier 2)
# Validate if GraphSAGE features are actually being used
plt.figure(figsize=(10, 8))
xgb.plot_importance(model_tier2, max_num_features=20, importance_type='gain', title='Feature Importance (Tier 2 Model)')
plt.tight_layout()
plt.show()

print("Analysis Complete.")