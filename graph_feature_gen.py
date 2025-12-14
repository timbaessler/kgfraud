import config
from graphdatascience import GraphDataScience
import pandas as pd
import os
import sys

# ==========================================
# 1. SETUP & CONNECTION
# ==========================================
try:
    gds = GraphDataScience(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    print(f"Connected to GDS: {gds.version()}")
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

# ==========================================
# 2. FEATURE ENRICHMENT (PRE-CALCULATION)
# ==========================================
# We calculate basic stats on nodes so GraphSAGE has 'raw material' to learn from.
print("Enriching nodes with transactional stats...")
gds.run_cypher("""
    MATCH (c:Customer)
    OPTIONAL MATCH (c)-[:MADE]->(t:Transaction)
    WITH c, avg(t.WARENKORB_WERT) as avg_val, count(t) as tx_cnt
    SET c.avg_val = coalesce(avg_val, 0.0), c.tx_cnt = coalesce(tx_cnt, 0)
""")

print("Enriching nodes with degree centrality...")
gds.run_cypher("""
    MATCH (c:Customer)
    OPTIONAL MATCH (c)--()
    WITH c, count(*) as degree
    SET c.degree = degree
""")

# ==========================================
# 3. GRAPH PROJECTION
# ==========================================
graph_name = "fraud-hybrid"
if gds.graph.exists(graph_name).exists:
    gds.graph.drop(graph_name)

print("Projecting graph (Memory)...")
# We use explicit default values ({defaultValue: 0}) to satisfy strict GDS requirements.
G, _ = gds.graph.project(
    graph_name,
    {
        "Customer": {
            "properties": {
                "degree": {"defaultValue": 0},
                "avg_val": {"defaultValue": 0.0},
                "tx_cnt": {"defaultValue": 0},
                "Fraud_incl_wo": {"defaultValue": 0}
            }
        },
        "Device": {
            "properties": {
                "degree": {"defaultValue": 0},
                "avg_val": {"defaultValue": 0.0},
                "tx_cnt": {"defaultValue": 0},
                "Fraud_incl_wo": {"defaultValue": 0}
            }
        },
        "Email": {
            "properties": {
                "degree": {"defaultValue": 0},
                "avg_val": {"defaultValue": 0.0},
                "tx_cnt": {"defaultValue": 0},
                "Fraud_incl_wo": {"defaultValue": 0}
            }
        },
        "IPAddress": {
            "properties": {
                "degree": {"defaultValue": 0},
                "avg_val": {"defaultValue": 0.0},
                "tx_cnt": {"defaultValue": 0},
                "Fraud_incl_wo": {"defaultValue": 0}
            }
        }
    },
    {"UNDIRECTED": {"type": "*", "orientation": "UNDIRECTED"}}
)
print(f"Graph projected: {G.node_count()} nodes, {G.relationship_count()} edges.")

# ==========================================
# 4. ALGORITHM A: WCC (COMMUNITY DETECTION)
# ==========================================
print("Running Weakly Connected Components (WCC)...")
gds.wcc.write(G, writeProperty="wccId")

# ==========================================
# 5. ALGORITHM B: GRAPHSAGE (EMBEDDINGS)
# ==========================================
print("Training GraphSAGE...")
model_name = "sage-model"

# Safe model cleanup
if gds.model.exists(model_name).exists:
    print(f"Dropping existing model: {model_name}")
    gds.model.drop(gds.model.get(model_name))

# Train the model
# Note: 'projectedNodeLabels' is removed as it is invalid in this GDS version
model, _ = gds.beta.graphSage.train(
    G,
    modelName=model_name,
    featureProperties=["degree", "avg_val", "tx_cnt"],
    embeddingDimension=64,
    learningRate=0.01,
    epochs=15,
    randomSeed=42
)

print("Writing GraphSAGE embeddings back to graph...")
gds.beta.graphSage.write(G, modelName=model_name, writeProperty="sage_emb")

# ==========================================
# 6. EXPORT FEATURES TO CSV
# ==========================================
print("Exporting features to CSV...")

# Corrected Cypher Query:
# 1. Matches Customers with embeddings.
# 2. Aggregates by WCC ID to calculate 'community_size'.
# 3. Unwinds the list to return one row per customer.
q = """
    MATCH (c:Customer)
    WHERE c.sage_emb IS NOT NULL
    WITH c.wccId as component, count(*) as community_size, collect(c) as members
    UNWIND members as member
    RETURN 
        member.GPID as customer_id,
        community_size,
        member.degree as degree_centrality,
        member.sage_emb as embedding
"""
df = gds.run_cypher(q)

if df.empty:
    print("❌ ERROR: No data found! GraphSAGE might have failed to write properties.")
else:
    # Post-Processing: Expand the embedding list into 64 separate columns
    print("Expanding embedding vectors...")
    emb_df = pd.DataFrame(df['embedding'].tolist(), index=df.index).add_prefix("sage_")

    # Merge scalar features with expanded embeddings
    final_df = pd.concat([df.drop(columns=['embedding']), emb_df], axis=1)

    output_file = "graph_features_final.csv"
    final_df.to_csv(output_file, index=False)

    print(f"✅ SUCCESS: Saved '{output_file}' with shape {final_df.shape}")
    print(f"Location: {os.path.abspath(output_file)}")

# ==========================================
# 7. CLEANUP
# ==========================================
print("Cleaning up GDS resources...")
gds.graph.drop(graph_name)
gds.model.drop(model)
print("Done.")