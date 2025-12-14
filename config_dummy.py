"""
Central configuration for the Fraud Detection Knowledge Graph.
"""

import os

# Neo4j Connection Details
# URI for Neo4j Desktop local instance
NEO4J_URI = "neo4j://127.0.0.1:7687"

# Authentication
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "YOURPASSWORD"  # <--- UPDATE THIS

# Database Name (Neo4j 5+ standard is usually 'neo4j')
NEO4J_DB = "neo4j"

# Data Source
# This filename must exist in the 'import' folder of your Neo4j DBMS
CSV_FILENAME = "transactions.csv"

# Path to the Neo4j Import Directory
NEO4J_IMPORT_DIR = "/Users/timbassler/Library/Application Support/neo4j-desktop/Application/Data/dbmss/dbms-d009b65b-4461-4867-a884-f7df01e971e1/import"
