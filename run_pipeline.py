"""
================================================================================
FRAUD DETECTION RESEARCH PIPELINE v4.0
================================================================================
Author: Tim Bäßler (Dissertation)
Date: January 2026

USAGE:
    python run_pipeline.py                 # Führt ALLE Experimente aus
    python run_pipeline.py --compare       # Zeigt Vergleichstabelle aller Ergebnisse

PIPELINE STEPS:
    1. Load/Transform Data → CSV
    2. Compute Diagnostics (Sharing, Super-nodes, Selectivity)
    3. Load Graph into Neo4j (kg_loader.py)
    4. Generate Graph Features (graph_feature_gen.py)
    5. Train & Evaluate (train_scientific.py)
    6. Report Results

EXPERIMENTS:
    - prop:        Proprietary Dataset (ThreatMetrix fingerprints)
    - prop_low:    Proprietary with degraded selectivity (truncate2)
    - ieee_naive:  IEEE-CIS with DeviceInfo+OS construction
    - ieee_medium: IEEE-CIS with card1+addr1 construction
    - ieee_kaggle: IEEE-CIS with Kaggle-UID construction
================================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import subprocess
import json
import hashlib
import zipfile
import argparse
import time
import gc
from datetime import datetime
from collections import Counter
from typing import Dict, Optional, List

# =============================================================================
# CONFIG
# =============================================================================

try:
    import config
    NEO4J_IMPORT_DIR = config.NEO4J_IMPORT_DIR
    PROP_CSV = config.CSV_FILENAME
    DATA_DIR = config.DATA_DIR
    RESULTS_DIR = config.RESULTS_DIR
    PROJECT_ROOT = config.PROJECT_ROOT
except ImportError:
    print("⚠️ config.py nicht gefunden - nutze Defaults")
    NEO4J_IMPORT_DIR = "."
    PROP_CSV = "part-00000-tid-788864549552509627-e9b4fc2e-1f09-426c-af7d-2f4e24ce4cd9-1555-1-c000.csv"
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    PROJECT_ROOT = os.getcwd()

# Verzeichnisse erstellen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️ Neo4j driver nicht verfügbar")

TARGET_DB = "neo4j"


# =============================================================================
# LOGGING
# =============================================================================

def log(message: str, level: str = "INFO"):
    icons = {
        "INFO": "🔹", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌",
        "STEP": "🚀", "HEADER": "📋", "METRIC": "📊", "WAIT": "⏳"
    }
    icon = icons.get(level, "🔹")
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icon} {message}")


# =============================================================================
# DIAGNOSTIC METRICS (Neue Paper-Metriken)
# =============================================================================

def compute_selectivity(series: pd.Series) -> Dict:
    """Berechnet Identifier Selectivity (Entropie-basiert)."""
    series = series.dropna()
    n_total = len(series)
    n_unique = series.nunique()
    
    if n_total == 0:
        return {"selectivity": 0.0, "n_unique": 0, "n_total": 0, "quality": "NO_DATA", "avg_cluster": 0}
    
    value_counts = series.value_counts()
    probabilities = value_counts / n_total
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    max_entropy = np.log2(n_total) if n_total > 1 else 0.0
    selectivity = entropy / max_entropy if max_entropy > 0 else 0.0
    
    if selectivity < 0.3: quality = "TOO_LOW"
    elif selectivity < 0.5: quality = "LOW"
    elif selectivity <= 0.8: quality = "OPTIMAL"
    elif selectivity <= 0.95: quality = "HIGH"
    else: quality = "TOO_HIGH"
    
    avg_cluster = n_total / n_unique if n_unique > 0 else 0
    
    return {
        "selectivity": round(selectivity, 4),
        "n_unique": n_unique,
        "n_total": n_total,
        "quality": quality,
        "avg_cluster": round(avg_cluster, 1)
    }


def compute_sharing_metrics(df: pd.DataFrame, device_col: str = 'TMX_DIGITAL_ID', 
                            customer_col: str = 'GPID') -> Dict:
    """Berechnet Device-Sharing Metriken."""
    device_sizes = df.groupby(device_col)[customer_col].nunique()
    
    total_devices = len(device_sizes)
    shared_devices = (device_sizes > 1).sum()
    shared_pct = shared_devices / total_devices if total_devices > 0 else 0
    
    max_customers = device_sizes.max() if len(device_sizes) > 0 else 0
    avg_shared = device_sizes[device_sizes > 1].mean() if shared_devices > 0 else 0
    
    return {
        "total_devices": total_devices,
        "shared_devices": int(shared_devices),
        "shared_pct": round(shared_pct * 100, 1),
        "max_customers_per_device": int(max_customers),
        "avg_customers_per_shared": round(avg_shared, 1)
    }


def compute_supernode_metrics(df: pd.DataFrame, device_col: str = 'TMX_DIGITAL_ID',
                               customer_col: str = 'GPID') -> Dict:
    """Berechnet Super-Node Metriken (Gini, Top-1 Konzentration)."""
    device_sizes = df.groupby(device_col)[customer_col].nunique().sort_values(ascending=False)
    total_customers = df[customer_col].nunique()
    
    if len(device_sizes) == 0 or total_customers == 0:
        return {"gini": 0, "top1_pct": 0, "top5_pct": 0, "failure_mode": "NO_DATA"}
    
    # Top-N Konzentration
    top1_pct = device_sizes.iloc[0] / total_customers * 100
    top5_pct = device_sizes.head(5).sum() / total_customers * 100
    
    # Gini-Koeffizient
    n = len(device_sizes)
    if n > 1:
        sorted_sizes = np.sort(device_sizes.values)
        cumsum = np.cumsum(sorted_sizes)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_sizes)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
    else:
        gini = 0
    
    # Failure Mode Detection
    if top1_pct > 50:
        failure_mode = "SUPERNODE_DOMINANCE"
    elif top1_pct > 10:
        failure_mode = "HIGH_CONCENTRATION"
    else:
        failure_mode = "OK"
    
    return {
        "gini": round(gini, 3),
        "top1_pct": round(top1_pct, 1),
        "top5_pct": round(top5_pct, 1),
        "failure_mode": failure_mode
    }


def run_full_diagnostics(df: pd.DataFrame, name: str = "Dataset") -> Dict:
    """Führt alle Diagnosen durch und gibt Empfehlung."""
    
    print(f"\n{'='*60}")
    log(f"DIAGNOSTICS: {name}", "HEADER")
    print('='*60)
    
    # 1. Selectivity
    sel = compute_selectivity(df['TMX_DIGITAL_ID'])
    log(f"Selectivity: {sel['selectivity']:.4f} ({sel['quality']})", "METRIC")
    log(f"   Unique Devices: {sel['n_unique']:,} / {sel['n_total']:,}", "INFO")
    
    # 2. Sharing
    sharing = compute_sharing_metrics(df)
    log(f"Sharing Rate: {sharing['shared_pct']:.1f}%", "METRIC")
    log(f"   Shared Devices: {sharing['shared_devices']:,} / {sharing['total_devices']:,}", "INFO")
    log(f"   Max Customers/Device: {sharing['max_customers_per_device']:,}", "INFO")
    
    # 3. Super-Nodes
    supernodes = compute_supernode_metrics(df)
    log(f"Gini Coefficient: {supernodes['gini']:.3f}", "METRIC")
    log(f"Top-1 Concentration: {supernodes['top1_pct']:.1f}%", "METRIC")
    
    # 4. Failure Mode Analysis
    failure_modes = []
    
    if sharing['shared_pct'] < 5:
        failure_modes.append("NO_SHARING")
        log("⚠️ FAILURE: <5% shared devices - Graph methods will NOT help!", "WARNING")
    
    if supernodes['top1_pct'] > 50:
        failure_modes.append("SUPERNODE_DOMINANCE")
        log(f"⚠️ FAILURE: {supernodes['top1_pct']:.1f}% in one device - Graph is a star!", "WARNING")
    elif supernodes['gini'] > 0.8:
        failure_modes.append("HIGH_INEQUALITY")
        log(f"⚠️ WARNING: Gini {supernodes['gini']:.3f} - Highly unequal distribution", "WARNING")
    
    if sharing['max_customers_per_device'] < 5 and sharing['shared_pct'] > 0:
        failure_modes.append("TOO_SPARSE")
        log(f"⚠️ WARNING: Max {sharing['max_customers_per_device']} customers/device - Too sparse", "WARNING")
    
    # 5. Recommendation
    if not failure_modes:
        recommendation = "GRAPH_SUITABLE"
        log("✅ RECOMMENDATION: Graph methods should provide lift!", "SUCCESS")
    else:
        recommendation = "GRAPH_NOT_SUITABLE"
        log(f"❌ RECOMMENDATION: Graph methods unlikely to help. Issues: {failure_modes}", "ERROR")
    
    return {
        "name": name,
        "selectivity": sel,
        "sharing": sharing,
        "supernodes": supernodes,
        "failure_modes": failure_modes,
        "recommendation": recommendation
    }


# =============================================================================
# DATABASE MANAGEMENT
# =============================================================================

def get_neo4j_driver():
    uri = config.NEO4J_URI.replace("neo4j://", "bolt://")
    return GraphDatabase.driver(uri, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))


def clear_database() -> bool:
    """Löscht alle Daten aus Neo4j."""
    if not NEO4J_AVAILABLE:
        return False
    
    log("Bereinige Datenbank...", "WAIT")
    driver = None
    
    try:
        driver = get_neo4j_driver()
        
        with driver.session(database=TARGET_DB) as session:
            # Count before
            count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
            
            if count == 0:
                log("DB ist bereits leer", "SUCCESS")
                return True
            
            log(f"   Lösche {count:,} Nodes...", "INFO")
            
            # Try APOC first
            try:
                session.run("""
                    CALL apoc.periodic.iterate(
                        "MATCH (n) RETURN n",
                        "DETACH DELETE n",
                        {batchSize: 10000, parallel: false}
                    )
                """).consume()
                time.sleep(2)
            except:
                # Fallback: Loop delete
                while True:
                    result = session.run("""
                        MATCH (n) WITH n LIMIT 50000 
                        DETACH DELETE n RETURN count(n) as c
                    """)
                    deleted = result.single()["c"]
                    if deleted == 0:
                        break
            
            # Verify
            final = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
            if final == 0:
                log("DB bereinigt", "SUCCESS")
                return True
            else:
                log(f"Noch {final:,} Nodes übrig!", "WARNING")
                return False
                
    except Exception as e:
        log(f"DB Fehler: {e}", "ERROR")
        return False
    finally:
        if driver:
            driver.close()


# =============================================================================
# DATA LOADING & TRANSFORMATION
# =============================================================================

def load_proprietary_data() -> Optional[pd.DataFrame]:
    """Lädt proprietäre Daten."""
    csv_path = os.path.join(NEO4J_IMPORT_DIR, PROP_CSV)
    if not os.path.exists(csv_path):
        csv_path = PROP_CSV
    
    if not os.path.exists(csv_path):
        log(f"Proprietäre CSV nicht gefunden: {csv_path}", "ERROR")
        return None
    
    log(f"Lade: {csv_path}", "INFO")
    df = pd.read_csv(csv_path, low_memory=False)
    log(f"Geladen: {len(df):,} Zeilen", "SUCCESS")
    return df


def load_ieee_data() -> Optional[pd.DataFrame]:
    """Lädt IEEE-CIS Daten."""
    data_dir = "benchmark_data"
    trans_path = os.path.join(data_dir, "train_transaction.csv")
    ident_path = os.path.join(data_dir, "train_identity.csv")
    
    if not os.path.exists(trans_path):
        log("Extrahiere ZIP...", "STEP")
        zip_path = "ieee-fraud-detection.zip"
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(data_dir)
        else:
            log(f"ZIP nicht gefunden: {zip_path}", "ERROR")
            return None
    
    log(f"Lade: {trans_path}", "INFO")
    df = pd.read_csv(trans_path, low_memory=False)
    
    if os.path.exists(ident_path):
        df_id = pd.read_csv(ident_path)
        df = df.merge(df_id, on="TransactionID", how="left")
        del df_id
        gc.collect()
    
    log(f"Geladen: {len(df):,} Zeilen", "SUCCESS")
    return df


def _apply_ieee_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """Wendet Standard-Mappings für IEEE an."""
    rename = {
        'TransactionID': 'RMS_PRUEF_ID',
        'isFraud': 'Fraud_incl_wo',
        'TransactionAmt': 'WARENKORB_WERT',
        'P_emaildomain': 'EMAIL'
    }
    df = df.rename(columns=rename)
    
    df['FIRST_NAME'] = 'Bench'
    df['LAST_NAME'] = 'Mark'
    df['KUNDENKLASSE'] = 'STD'
    df['EMC_SCORE_FINAL'] = 0
    df['TMX_DEV_POLICY_SCORE'] = 0
    df['TMX_DEV_IPADDRESS'] = df['addr1'].fillna('0').astype(str)
    
    for c in ['LA_STRASSE', 'LA_HAUSNUMMER', 'LA_PLZ', 'LA_ORT']:
        df[c] = ''
    
    start_ts = datetime(2017, 12, 1).timestamp()
    df['PRUEFUNG_INTERN_DATUM'] = df['TransactionDT'].apply(
        lambda x: datetime.fromtimestamp(start_ts + x).isoformat()
    )
    
    return df


def transform_ieee_naive(df: pd.DataFrame) -> pd.DataFrame:
    """IEEE NAIVE: DeviceInfo + OS → Device ID."""
    log("Transformation: IEEE NAIVE", "STEP")
    
    df['device_fp'] = (df['DeviceInfo'].fillna('UNK').astype(str) + '_' + 
                       df['id_30'].fillna('UNK').astype(str))
    df['TMX_DIGITAL_ID'] = df['device_fp'].apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])
    df['GPID'] = 'CUST_' + df['card1'].astype(str) + '_' + df['card2'].fillna('UNK').astype(str)
    
    return _apply_ieee_mappings(df)


def transform_ieee_medium(df: pd.DataFrame) -> pd.DataFrame:
    """IEEE MEDIUM: card1 + addr1 → Device ID."""
    log("Transformation: IEEE MEDIUM", "STEP")
    
    device_raw = (df['card1'].fillna(-1).astype(int).astype(str) + '_' + 
                  df['addr1'].fillna(-1).astype(int).astype(str))
    df['TMX_DIGITAL_ID'] = device_raw.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])
    df['GPID'] = 'CUST_' + df['card1'].astype(str) + '_' + df['addr1'].fillna(-1).astype(int).astype(str)
    
    return _apply_ieee_mappings(df)


def transform_ieee_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """IEEE KAGGLE: card1 + addr1 + email_domain → Device ID."""
    log("Transformation: IEEE KAGGLE-UID", "STEP")
    
    device_raw = (df['card1'].fillna(-1).astype(int).astype(str) + '_' + 
                  df['addr1'].fillna(-1).astype(int).astype(str) + '_' +
                  df['P_emaildomain'].fillna('UNK').astype(str))
    df['TMX_DIGITAL_ID'] = device_raw.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])
    df['GPID'] = 'CUST_' + df['card1'].astype(str) + '_' + df['card2'].fillna('UNK').astype(str)
    
    return _apply_ieee_mappings(df)


def transform_prop_low(df: pd.DataFrame, strategy: str = "truncate2") -> pd.DataFrame:
    """Proprietary mit degradierter Selectivity."""
    log(f"Transformation: PROP_LOW ({strategy})", "STEP")
    
    if strategy == "truncate2":
        df['TMX_DIGITAL_ID'] = df['TMX_DIGITAL_ID'].astype(str).str[:2]
    elif strategy == "truncate4":
        df['TMX_DIGITAL_ID'] = df['TMX_DIGITAL_ID'].astype(str).str[:4]
    elif strategy == "bucket100":
        df['TMX_DIGITAL_ID'] = df['TMX_DIGITAL_ID'].apply(
            lambda x: f"BUCKET_{hash(str(x)) % 100:03d}"
        )
    elif strategy == "bucket500":
        df['TMX_DIGITAL_ID'] = df['TMX_DIGITAL_ID'].apply(
            lambda x: f"BUCKET_{hash(str(x)) % 500:03d}"
        )
    else:
        log(f"Unbekannte Strategie: {strategy}", "ERROR")
    
    return df


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_pipeline_step(script: str, env: dict, timeout: int = 7200) -> bool:
    """Führt ein Pipeline-Script aus."""
    log(f"Starte: {script}", "STEP")
    
    gc.collect()
    
    try:
        full_env = os.environ.copy()
        full_env.update(env)
        
        result = subprocess.run(
            [sys.executable, script],
            env=full_env,
            timeout=timeout
        )
        
        if result.returncode == 0:
            log(f"{script} erfolgreich", "SUCCESS")
            return True
        else:
            log(f"{script} fehlgeschlagen (Code {result.returncode})", "ERROR")
            return False
            
    except subprocess.TimeoutExpired:
        log(f"{script} Timeout nach {timeout}s", "ERROR")
        return False
    except Exception as e:
        log(f"{script} Exception: {e}", "ERROR")
        return False


def run_experiment(dataset: str) -> Dict:
    """Führt ein komplettes Experiment durch."""
    
    print("\n" + "="*70)
    log(f"EXPERIMENT: {dataset.upper()}", "HEADER")
    print("="*70)
    
    # 1. Daten laden
    if dataset == "prop":
        df = load_proprietary_data()
        csv_file = "proprietary_transformed.csv"
    elif dataset == "prop_low":
        df = load_proprietary_data()
        if df is not None:
            df = transform_prop_low(df, strategy="truncate2")
        csv_file = "prop_low_truncate2_transformed.csv"
    elif dataset.startswith("ieee_"):
        df = load_ieee_data()
        construction = dataset.replace("ieee_", "")
        
        if df is not None:
            if construction == "naive":
                df = transform_ieee_naive(df)
            elif construction == "medium":
                df = transform_ieee_medium(df)
            elif construction == "kaggle":
                df = transform_ieee_kaggle(df)
            else:
                log(f"Unbekannte IEEE Konstruktion: {construction}", "ERROR")
                return {"error": f"Unknown construction: {construction}"}
        
        csv_file = f"ieee_{construction}_transformed.csv"
    else:
        log(f"Unbekanntes Dataset: {dataset}", "ERROR")
        return {"error": f"Unknown dataset: {dataset}"}
    
    if df is None:
        return {"error": "Data loading failed"}
    
    # 2. Diagnostics
    diagnostics = run_full_diagnostics(df, dataset)
    
    # 3. CSV speichern (in data/ Verzeichnis)
    csv_file = os.path.join(DATA_DIR, csv_file)
    df.to_csv(csv_file, index=False)
    log(f"CSV gespeichert: {csv_file}", "SUCCESS")
    
    # RAM freigeben
    del df
    gc.collect()
    
    # 4. Graph Pipeline
    if not clear_database():
        return {"error": "Database clear failed", "diagnostics": diagnostics}
    
    time.sleep(3)
    
    results_file = os.path.join(RESULTS_DIR, f"{dataset}_results.json")
    env = {
        "CSV_FILENAME": os.path.basename(csv_file),  # Nur Dateiname, kg_loader sucht selbst
        "NEO4J_DB": TARGET_DB,
        "RESULTS_FILENAME": results_file
    }
    
    # Pipeline Steps
    steps = [
        ("kg_loader.py", "Graph Import"),
        ("graph_feature_gen.py", "Feature Generation"),
        ("train_scientific.py", "Training & Evaluation")
    ]
    
    for script, desc in steps:
        if not os.path.exists(script):
            log(f"Script nicht gefunden: {script}", "ERROR")
            return {"error": f"Script missing: {script}", "diagnostics": diagnostics}
        
        if not run_pipeline_step(script, env):
            return {"error": f"Pipeline failed at {script}", "diagnostics": diagnostics}
    
    # 5. Ergebnisse laden
    results = {"dataset": dataset, "diagnostics": diagnostics}
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            ml_results = json.load(f)
        
        baseline = ml_results.get('Baseline', {}).get('pr_mean', 0)
        full = ml_results.get('Full', {}).get('pr_mean', baseline)
        lift = (full - baseline) * 100 if baseline > 0 else 0
        
        results['baseline_pr_auc'] = round(baseline, 4)
        results['full_pr_auc'] = round(full, 4)
        results['lift_pct'] = round(lift, 2)
        
        log(f"RESULT: Baseline={baseline:.4f}, Full={full:.4f}, Lift={lift:+.2f}%", "SUCCESS")
    
    return results


def print_comparison_table(all_results: List[Dict]):
    """Druckt Vergleichstabelle."""
    print("\n" + "="*90)
    print("EXPERIMENT COMPARISON")
    print("="*90)
    print(f"{'Dataset':<15} | {'Select.':<8} | {'Shared%':<8} | {'Gini':<6} | {'Top1%':<7} | {'Baseline':<8} | {'Full':<8} | {'Lift':<8}")
    print("-"*90)
    
    for r in all_results:
        if "error" in r:
            print(f"{r.get('dataset', '?'):<15} | ERROR: {r['error']}")
            continue
        
        d = r.get('diagnostics', {})
        sel = d.get('selectivity', {}).get('selectivity', 0)
        shared = d.get('sharing', {}).get('shared_pct', 0)
        gini = d.get('supernodes', {}).get('gini', 0)
        top1 = d.get('supernodes', {}).get('top1_pct', 0)
        
        baseline = r.get('baseline_pr_auc', 0)
        full = r.get('full_pr_auc', 0)
        lift = r.get('lift_pct', 0)
        
        print(f"{r['dataset']:<15} | {sel:<8.4f} | {shared:<8.1f} | {gini:<6.3f} | {top1:<7.1f} | {baseline:<8.4f} | {full:<8.4f} | {lift:>+7.2f}%")
    
    print("="*90)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Research Pipeline v4.0")
    
    parser.add_argument('--dataset', type=str, 
                        choices=['prop', 'prop_low', 'ieee_naive', 'ieee_medium', 'ieee_kaggle'],
                        help='Dataset to run experiment on')
    parser.add_argument('--diagnose', type=str, metavar='CSV_FILE',
                        help='Run diagnostics only on a CSV file')
    parser.add_argument('--all-ieee', action='store_true',
                        help='Run all IEEE experiments')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all existing results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("   FRAUD DETECTION RESEARCH PIPELINE v4.0")
    print("   Tim Bäßler - Dissertation 2026")
    print("="*70)
    
    # Diagnose-Only Mode
    if args.diagnose:
        if not os.path.exists(args.diagnose):
            log(f"Datei nicht gefunden: {args.diagnose}", "ERROR")
            sys.exit(1)
        
        df = pd.read_csv(args.diagnose, low_memory=False)
        run_full_diagnostics(df, args.diagnose)
        sys.exit(0)
    
    # Compare Mode
    if args.compare:
        all_results = []
        for dataset in ['prop', 'prop_low', 'ieee_naive', 'ieee_medium', 'ieee_kaggle']:
            results_file = os.path.join(RESULTS_DIR, f"{dataset}_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    ml_results = json.load(f)
                
                # Load diagnostics if available
                diag_file = os.path.join(RESULTS_DIR, f"{dataset}_diagnostics.json")
                diagnostics = {}
                if os.path.exists(diag_file):
                    with open(diag_file, 'r') as f:
                        diagnostics = json.load(f)
                
                baseline = ml_results.get('Baseline', {}).get('pr_mean', 0)
                full = ml_results.get('Full', {}).get('pr_mean', baseline)
                lift = (full - baseline) * 100 if baseline > 0 else 0
                
                all_results.append({
                    'dataset': dataset,
                    'diagnostics': diagnostics,
                    'baseline_pr_auc': baseline,
                    'full_pr_auc': full,
                    'lift_pct': lift
                })
        
        if all_results:
            print_comparison_table(all_results)
        else:
            log("Keine Ergebnisse gefunden. Führe erst Experimente aus.", "WARNING")
        sys.exit(0)
    
    # All IEEE Mode
    if args.all_ieee:
        all_results = []
        
        for dataset in ['ieee_naive', 'ieee_medium', 'ieee_kaggle']:
            result = run_experiment(dataset)
            all_results.append(result)
            
            # Save diagnostics
            if 'diagnostics' in result:
                diag_file = os.path.join(RESULTS_DIR, f"{dataset}_diagnostics.json")
                with open(diag_file, 'w') as f:
                    json.dump(result['diagnostics'], f, indent=2)
            
            log("Cooling down (30s)...", "WAIT")
            time.sleep(30)
        
        print_comparison_table(all_results)
        
        # Export summary
        summary_file = os.path.join(RESULTS_DIR, "ieee_comparison_results.json")
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        log(f"Ergebnisse in {summary_file}", "SUCCESS")
        sys.exit(0)
    
    # Single Dataset Mode
    if args.dataset:
        result = run_experiment(args.dataset)
        
        # Save diagnostics
        if 'diagnostics' in result:
            diag_file = os.path.join(RESULTS_DIR, f"{args.dataset}_diagnostics.json")
            with open(diag_file, 'w') as f:
                json.dump(result['diagnostics'], f, indent=2)
        
        print_comparison_table([result])
        sys.exit(0)
    
    # No arguments - RUN EVERYTHING
    print("\n" + "="*70)
    log("RUNNING FULL EXPERIMENT SUITE", "HEADER")
    print("="*70)
    
    all_results = []
    
    # 1. Proprietary Dataset
    log("Starting Proprietary Dataset...", "STEP")
    result = run_experiment("prop")
    all_results.append(result)
    if 'diagnostics' in result:
        diag_file = os.path.join(RESULTS_DIR, "prop_diagnostics.json")
        with open(diag_file, 'w') as f:
            json.dump(result['diagnostics'], f, indent=2)
    
    log("Cooling down (30s)...", "WAIT")
    time.sleep(30)
    
    # 2. All IEEE Variants
    for dataset in ['ieee_naive', 'ieee_medium', 'ieee_kaggle']:
        log(f"Starting {dataset}...", "STEP")
        result = run_experiment(dataset)
        all_results.append(result)
        
        if 'diagnostics' in result:
            diag_file = os.path.join(RESULTS_DIR, f"{dataset}_diagnostics.json")
            with open(diag_file, 'w') as f:
                json.dump(result['diagnostics'], f, indent=2)
        
        log("Cooling down (30s)...", "WAIT")
        time.sleep(30)
    
    # 3. Proprietary LOW (degraded selectivity)
    log("Starting Proprietary LOW...", "STEP")
    result = run_experiment("prop_low")
    all_results.append(result)
    if 'diagnostics' in result:
        diag_file = os.path.join(RESULTS_DIR, "prop_low_diagnostics.json")
        with open(diag_file, 'w') as f:
            json.dump(result['diagnostics'], f, indent=2)
    
    # Final Summary
    print_comparison_table(all_results)
    
    # Export all results
    summary_file = os.path.join(RESULTS_DIR, "all_experiments_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    log(f"Alle Ergebnisse gespeichert in {summary_file}", "SUCCESS")
    print("\n🎉 FERTIG! Alle Experimente abgeschlossen.")


if __name__ == "__main__":
    main()
