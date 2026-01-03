"""
================================================================================
GRAPH FEATURE GENERATION v2.1
================================================================================
- Respektiert neue Ordnerstruktur (data/, results/)
- graph_features_comparison.csv wird im Projekt-Root gespeichert
- Kompatibel mit Neo4j 5.x
================================================================================
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Config importieren
try:
    import config
    DATA_DIR = config.DATA_DIR
    RESULTS_DIR = config.RESULTS_DIR
    PROJECT_ROOT = config.PROJECT_ROOT
except ImportError:
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    PROJECT_ROOT = os.getcwd()
    
    class config:
        NEO4J_URI = "bolt://127.0.0.1:7687"
        NEO4J_USER = "neo4j"
        NEO4J_PASSWORD = "password"
        NEO4J_DB = "neo4j"

from graphdatascience import GraphDataScience

# ==========================================
# CONFIGURATION
# ==========================================
SOFT_SUPERNODE_THRESHOLD = 200
SUPERNODE_WEIGHT_DECAY = 0.1

EMBEDDING_DIM = 64
NODE2VEC_DIM = 32

NODE2VEC_P = 0.5
NODE2VEC_Q = 2.0
NODE2VEC_WALK_LENGTH = 40
NODE2VEC_WALKS_PER_NODE = 10

TRUNCATION_RATIO = 0.5

MAX_GRAPH_NODES_FOR_NODE2VEC = 1_000_000
USE_NATIVE_PROJECTION = True

# Output file - im Projekt-Root für Kompatibilität mit train_scientific.py
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "graph_features_comparison.csv")


# ==========================================
# 1. SETUP & CONNECTION
# ==========================================
try:
    DB_NAME = os.environ.get('NEO4J_DB', config.NEO4J_DB if hasattr(config, 'NEO4J_DB') else 'neo4j')
    gds = GraphDataScience(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        database=DB_NAME
    )
    print(f"✅ Connected to GDS: {gds.version()} (Database: {DB_NAME})")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    sys.exit(1)


# ==========================================
# 2. DATE BOUNDARY CALCULATION
# ==========================================
print("⏳ Calculating time boundaries...")
try:
    time_res = gds.run_cypher("""
        MATCH (t:Transaction)
        RETURN toString(min(t.date)) as min_date, toString(max(t.date)) as max_date
    """).iloc[0]
    
    min_dt = pd.to_datetime(time_res['min_date'])
    max_dt = pd.to_datetime(time_res['max_date'])
    
    total_duration = max_dt - min_dt
    cutoff_dt = min_dt + (total_duration * TRUNCATION_RATIO)
    
    baseline_date_str = "1970-01-01T00:00:00"
    trunc_date_str = cutoff_dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    print(f"   → Data Range: {min_dt} to {max_dt}")
    print(f"   → Truncation Point ({TRUNCATION_RATIO*100}%): {cutoff_dt}")
    
except Exception as e:
    print(f"❌ Failed to calculate dates: {e}")
    sys.exit(1)


# ==========================================
# 3. SUPERNODE WEIGHT COMPUTATION
# ==========================================
print("\n🔧 Computing Supernode Weights...")
try:
    degree_query = """
    MATCH (n)
    WHERE n:Device OR n:Email OR n:IPAddress OR n:Address
    WITH n, COUNT { (n)--() } as degree
    SET n.raw_degree = degree,
        n.edge_weight = CASE 
            WHEN degree <= $threshold THEN 1.0
            ELSE 1.0 / (1.0 + $decay * (degree - $threshold))
        END
    RETURN labels(n)[0] as label, 
           count(n) as node_count,
           avg(degree) as avg_degree, 
           max(degree) as max_degree,
           avg(n.edge_weight) as avg_weight
    """
    weight_stats = gds.run_cypher(degree_query, {
        'threshold': SOFT_SUPERNODE_THRESHOLD,
        'decay': SUPERNODE_WEIGHT_DECAY
    })
    print("   Supernode Weight Statistics:")
    print(weight_stats.to_string(index=False))
except Exception as e:
    print(f"   ⚠️ Weight computation skipped: {e}")


# ==========================================
# 4. NATIVE PROJECTION FUNCTION
# ==========================================
def create_native_projection(graph_name, undirected=False):
    """Creates a native projection."""
    try:
        gds.graph.drop(graph_name)
    except:
        pass
    
    node_labels = ['Customer', 'Transaction', 'Device', 'Email', 'IPAddress', 'Address']
    orientation = "UNDIRECTED" if undirected else "NATURAL"
    
    rel_types = {
        'MADE': {'orientation': orientation},
        'PERFORMED': {'orientation': orientation},
        'USED_DEVICE': {'orientation': orientation},
        'USES_EMAIL': {'orientation': orientation},
        'FROM_IP': {'orientation': orientation},
        'SHIPPED_TO': {'orientation': orientation},
        'PLACED_FROM': {'orientation': orientation}
    }
    
    print(f"   → Creating native projection '{graph_name}' (undirected={undirected})...")
    
    G, result = gds.graph.project(graph_name, node_labels, rel_types)
    return G


# ==========================================
# 5. FEATURE PIPELINE
# ==========================================
def run_feature_pipeline_v3(suffix, min_date_string, use_node2vec=True):
    """Enhanced feature pipeline."""
    graph_name = f"graph{suffix}"
    graph_name_undirected = f"graph{suffix}_undirected"
    
    print(f"\n🚀 --- START PIPELINE: {suffix} ---")
    
    # STEP A: Create DIRECTED projection
    try:
        G = create_native_projection(graph_name, undirected=False)
        node_count = G.node_count()
        edge_count = G.relationship_count()
        print(f"   ✅ Projected (directed): {node_count:,} nodes, {edge_count:,} edges")
    except Exception as e:
        print(f"   ❌ Native projection failed: {e}")
        return

    # STEP B: Degree Centrality
    print(f"   → Calculating Degree Centrality")
    try:
        gds.degree.write(G, writeProperty=f'degree{suffix}')
        print(f"      ✅ Degree complete")
    except Exception as e:
        print(f"      ❌ Degree failed: {e}")

    # STEP C: Weakly Connected Components
    print(f"   → Calculating WCC")
    try:
        gds.wcc.write(G, writeProperty=f'wccId{suffix}')
        print(f"      ✅ WCC complete")
    except Exception as e:
        print(f"      ❌ WCC failed: {e}")

    # STEP D: FastRP Embeddings
    print(f"   → Calculating FastRP Embeddings (dim={EMBEDDING_DIM})")
    try:
        gds.fastRP.write(
            G, 
            embeddingDimension=EMBEDDING_DIM,
            iterationWeights=[1.0, 1.0, 1.0],
            writeProperty=f'embedding{suffix}'
        )
        print(f"      ✅ FastRP complete")
    except Exception as e:
        print(f"      ❌ FastRP failed: {e}")
        gds.run_cypher(f"MATCH (c:Customer) SET c.embedding{suffix} = null")

    # Clean up directed graph
    try:
        gds.graph.drop(graph_name)
    except:
        pass

    # STEP E: UNDIRECTED projection for Triangle/LCC
    print(f"   → Creating UNDIRECTED projection for Triangle Count & LCC")
    try:
        G_undirected = create_native_projection(graph_name_undirected, undirected=True)
        print(f"   ✅ Projected (undirected): {G_undirected.node_count():,} nodes")
        
        # Triangle Count
        print(f"   → Calculating Triangle Count")
        try:
            gds.triangleCount.write(G_undirected, writeProperty=f'triangles{suffix}')
            print(f"      ✅ Triangle Count complete")
        except Exception as e:
            print(f"      ⚠️ Triangle Count failed: {e}")
            gds.run_cypher(f"MATCH (c:Customer) SET c.triangles{suffix} = 0")
        
        # Local Clustering Coefficient
        print(f"   → Calculating Local Clustering Coefficient")
        try:
            gds.localClusteringCoefficient.write(G_undirected, writeProperty=f'lcc{suffix}')
            print(f"      ✅ LCC complete")
        except Exception as e:
            print(f"      ⚠️ LCC failed: {e}")
            gds.run_cypher(f"MATCH (c:Customer) SET c.lcc{suffix} = 0.0")
        
        gds.graph.drop(graph_name_undirected)
        
    except Exception as e:
        print(f"   ⚠️ Undirected projection failed: {e}")
        gds.run_cypher(f"MATCH (c:Customer) SET c.triangles{suffix} = 0, c.lcc{suffix} = 0.0")

    # STEP F: Node2Vec (Optional)
    if use_node2vec and node_count < MAX_GRAPH_NODES_FOR_NODE2VEC:
        print(f"   → Calculating Node2Vec")
        try:
            G_n2v = create_native_projection(f"{graph_name}_n2v", undirected=False)
            gds.node2vec.write(
                G_n2v, 
                embeddingDimension=NODE2VEC_DIM,
                walkLength=NODE2VEC_WALK_LENGTH, 
                walksPerNode=NODE2VEC_WALKS_PER_NODE,
                returnFactor=NODE2VEC_P, 
                inOutFactor=NODE2VEC_Q,
                writeProperty=f'node2vec{suffix}'
            )
            gds.graph.drop(f"{graph_name}_n2v")
            print(f"      ✅ Node2Vec complete")
        except Exception as e:
            print(f"      ⚠️ Node2Vec failed: {e}")
            gds.run_cypher(f"MATCH (c:Customer) SET c.node2vec{suffix} = null")
    else:
        gds.run_cypher(f"MATCH (c:Customer) SET c.node2vec{suffix} = null")

    print(f"   🎉 Pipeline {suffix} complete.")


# ==========================================
# 6. EXECUTION
# ==========================================
print("\n" + "="*60)
print("EXECUTING FEATURE PIPELINES")
print("="*60)

run_feature_pipeline_v3("_full", baseline_date_str, use_node2vec=False)
run_feature_pipeline_v3("_trunc", trunc_date_str, use_node2vec=False)


# ==========================================
# 7. EXPORT
# ==========================================
print("\n💾 Exporting features...")

query = """
    MATCH (c:Customer)
    WHERE c.embedding_full IS NOT NULL 
    RETURN 
        c.GPID as customer_id, 
        c.risk_score as risk_score,
        c.degree_full as degree_full, 
        c.wccId_full as wccId_full,
        c.embedding_full as emb_full,
        coalesce(c.triangles_full, 0) as triangles_full,
        coalesce(c.lcc_full, 0.0) as lcc_full,
        c.node2vec_full as node2vec_full,
        coalesce(c.degree_trunc, 0) as degree_trunc,
        coalesce(c.wccId_trunc, -1) as wccId_trunc,
        c.embedding_trunc as emb_trunc,
        coalesce(c.triangles_trunc, 0) as triangles_trunc,
        coalesce(c.lcc_trunc, 0.0) as lcc_trunc
"""

df = gds.run_cypher(query)

if df.empty:
    print("⚠️ Result is empty. Trying fallback query...")
    query_fallback = """
        MATCH (c:Customer)
        RETURN 
            c.GPID as customer_id, 
            c.risk_score as risk_score,
            coalesce(c.degree_full, 0) as degree_full, 
            coalesce(c.wccId_full, -1) as wccId_full,
            c.embedding_full as emb_full,
            coalesce(c.triangles_full, 0) as triangles_full,
            coalesce(c.lcc_full, 0.0) as lcc_full,
            c.node2vec_full as node2vec_full,
            coalesce(c.degree_trunc, 0) as degree_trunc,
            coalesce(c.wccId_trunc, -1) as wccId_trunc,
            c.embedding_trunc as emb_trunc,
            coalesce(c.triangles_trunc, 0) as triangles_trunc,
            coalesce(c.lcc_trunc, 0.0) as lcc_trunc
    """
    df = gds.run_cypher(query_fallback)
    
if df.empty:
    print("❌ Error: No customers found in database.")
    sys.exit(1)

print(f"   → Fetched {len(df):,} records. Expanding embeddings...")


def expand_embeddings(dataframe, col_name, prefix, dim):
    """Safely expand embedding columns."""
    empty_vec = [0.0] * dim
    
    def safe_expand(x):
        if x is None:
            return empty_vec
        if isinstance(x, list) and len(x) == dim:
            return x
        return empty_vec
    
    safe_series = dataframe[col_name].apply(safe_expand)
    emb_df = pd.DataFrame(safe_series.tolist(), index=dataframe.index)
    emb_df.columns = [f"{prefix}_{i}" for i in range(dim)]
    return emb_df


# Expand embeddings
df_fastrp_full = expand_embeddings(df, 'emb_full', 'fastrp_full', EMBEDDING_DIM)
df_fastrp_trunc = expand_embeddings(df, 'emb_trunc', 'fastrp_trunc', EMBEDDING_DIM)
df_n2v_full = expand_embeddings(df, 'node2vec_full', 'n2v_full', NODE2VEC_DIM)

# Build final dataframe
final_df = pd.concat([
    df.drop(columns=['emb_full', 'emb_trunc', 'node2vec_full'], errors='ignore'),
    df_fastrp_full, 
    df_fastrp_trunc, 
    df_n2v_full
], axis=1)

# Speichern
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ SUCCESS: Saved '{OUTPUT_FILE}'")
print(f"   → Shape: {final_df.shape}")
print(f"   → Columns: {list(final_df.columns[:10])}... (+{len(final_df.columns)-10} more)")
print("   → Ready for train_scientific.py!")
