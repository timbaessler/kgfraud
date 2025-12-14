import logging
import sys
from neo4j import GraphDatabase
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_query(tx, query, params=None):
    result = tx.run(query, params)
    return result.consume()

def main():
    logger.info("Connecting to Neo4j...")

    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        driver.verify_connectivity()
        logger.info(f"Connected to {config.NEO4J_URI}")
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        sys.exit(1)

    with driver.session(database=config.NEO4J_DB) as session:

        # 1. Clean Slate (Reset DB)
        logger.info("Clearing database...")
        session.run("MATCH (n) DETACH DELETE n")

        # 2. Create Constraints (Crucial for import performance and data integrity)
        logger.info("Creating uniqueness constraints...")
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer) REQUIRE c.GPID IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Email) REQUIRE e.addr IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Device) REQUIRE d.fingerprint IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:IPAddress) REQUIRE i.addr IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Address) REQUIRE a.addrHash IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE"
        ]

        for const in constraints:
            session.run(const)

        # 3. Load Data
        # We use LOAD CSV. Note: The file must be in the Neo4j 'import' directory.
        logger.info(f"Loading data from {config.CSV_FILENAME}...")

        load_query = f"""
        LOAD CSV WITH HEADERS FROM 'file:///{config.CSV_FILENAME}' AS line
        
        // 1. Create Customer
        MERGE (c:Customer {{GPID: line.GPID}})
        SET c.Fraud_incl_wo = toInteger(line.Fraud_incl_wo)
        
        // 2. Create Email and link
        MERGE (e:Email {{addr: line.EMAIL}})
        MERGE (c)-[:USES_EMAIL]->(e)
        
        // 3. Create Device and link
        MERGE (d:Device {{fingerprint: line.TMX_DIGITAL_ID}})
        MERGE (c)-[:USED_DEVICE]->(d)
        
        // 4. Create IP and link Device to IP (Device placed from IP)
        MERGE (ip:IPAddress {{addr: line.TMX_DEV_IPADDRESS}})
        MERGE (d)-[:PLACED_FROM]->(ip)
        
        // 5. Create Address (Composite Hash) and link
        WITH c, line, 
             line.LA_STRASSE + '|' + line.LA_HAUSNUMMER + '|' + line.LA_PLZ + '|' + line.LA_ORT AS addressHash
        MERGE (a:Address {{addrHash: addressHash}})
        MERGE (c)-[:SHIPPED_TO]->(a)
        
        // 6. Create Transaction (1 row = 1 transaction)
        // We use linenumber() or a UUID approach if tx_id isn't in CSV. 
        // Here we assume generating an ID based on properties or row id if needed, 
        // but let's assume unique combination or synthesize a UUID.
        MERGE (t:Transaction {{tx_id: linenumber()}})
        SET t.Fraud_incl_wo = toInteger(line.Fraud_incl_wo)
        MERGE (c)-[:MADE]->(t)
        """

        # Execute the load
        summary = session.run(load_query).consume()

        logger.info(f"Import complete.")
        logger.info(f"Nodes created: {summary.counters.nodes_created}")
        logger.info(f"Relationships created: {summary.counters.relationships_created}")

    driver.close()

if __name__ == "__main__":
    main()