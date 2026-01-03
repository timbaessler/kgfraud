"""
================================================================================
FRAUD GRAPH LOADER v2.1
================================================================================
- Respektiert neue Ordnerstruktur (data/, results/)
- Environment Variable CSV_FILENAME hat Priorität
- Automatisches Kopieren in Neo4j Import Directory
================================================================================
"""

import logging
import sys
import os
import time
import shutil
from neo4j import GraphDatabase
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class FraudGraphLoader:
    def __init__(self):
        self.driver = None
        
        # CSV Filename: ENV > config
        self.csv_filename = os.environ.get('CSV_FILENAME', config.CSV_FILENAME)
        logger.info(f"📄 CSV Source: {self.csv_filename}")

    def connect(self):
        uri = config.NEO4J_URI
        if "neo4j://" in uri and "localhost" in uri:
            uri = uri.replace("neo4j://", "bolt://")

        retries = 3
        for i in range(retries):
            try:
                self.driver = GraphDatabase.driver(
                    uri,
                    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                    max_connection_lifetime=60,
                    max_connection_pool_size=1,
                    connection_acquisition_timeout=30.0,
                    liveness_check_timeout=0,
                    keep_alive=True
                )
                self.driver.verify_connectivity()
                logger.info(f"✅ Connected to Neo4j at {uri}")
                return
            except Exception as e:
                logger.warning(f"Connection attempt {i+1} failed: {e}")
                time.sleep(2)
        
        logger.error("❌ Could not connect to Neo4j")
        sys.exit(1)

    def wait_for_database(self):
        target_db = os.environ.get("NEO4J_DB", config.NEO4J_DB)
        logger.info(f"⏳ Checking database '{target_db}'...")
        
        max_retries = 45
        for i in range(max_retries):
            try:
                with self.driver.session(database="system") as session:
                    result = session.run("SHOW DATABASES YIELD name, currentStatus").data()
                    db_info = next((db for db in result if db['name'] == target_db), None)
                
                if not db_info:
                    logger.error(f"❌ Database '{target_db}' not found!")
                    sys.exit(1)
                
                status = db_info['currentStatus']
                
                if status == 'online':
                    logger.info(f"✅ Database '{target_db}' is ONLINE")
                    return
                elif status == 'offline':
                    logger.warning(f"Database is OFFLINE, starting...")
                    with self.driver.session(database="system") as session:
                        session.run(f"START DATABASE {target_db} WAIT 60").consume()
                else:
                    logger.info(f"   Status: {status}... (Attempt {i+1})")
                
                time.sleep(2)
            except Exception as e:
                logger.warning(f"   Waiting... ({e})")
                time.sleep(2)
        
        logger.error("❌ TIMEOUT: Database stuck")
        sys.exit(1)

    def close(self):
        if self.driver:
            self.driver.close()

    def create_constraints(self):
        logger.info("Creating constraints...")
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer) REQUIRE c.GPID IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Email) REQUIRE e.addr IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Device) REQUIRE d.fingerprint IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:IPAddress) REQUIRE i.addr IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Address) REQUIRE a.addrHash IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (t:Transaction) ON (t.date)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Customer) ON (c.fraud_label)"
        ]
        db_name = os.environ.get("NEO4J_DB", config.NEO4J_DB)
        with self.driver.session(database=db_name) as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.warning(f"Constraint notice: {e}")

    def ensure_csv_in_import_dir(self) -> str:
        """
        Stellt sicher, dass die CSV im Neo4j Import-Verzeichnis liegt.
        Sucht in: data/, NEO4J_IMPORT_DIR, current dir
        """
        csv_basename = os.path.basename(self.csv_filename)
        import_dir = config.NEO4J_IMPORT_DIR
        target_path = os.path.join(import_dir, csv_basename)
        
        # Bereits vorhanden?
        if os.path.exists(target_path):
            logger.info(f"✅ CSV already in import dir: {csv_basename}")
            return csv_basename
        
        # Suche in verschiedenen Locations
        search_paths = [
            self.csv_filename,                              # Exakter Pfad
            os.path.join(config.DATA_DIR, csv_basename),    # data/ Verzeichnis
            os.path.join(config.DATA_DIR, self.csv_filename),
            os.path.join(os.getcwd(), self.csv_filename),   # Current dir
            os.path.join(os.getcwd(), csv_basename),
            os.path.join(config.PROJECT_ROOT, self.csv_filename),
            os.path.join(config.PROJECT_ROOT, csv_basename),
        ]
        
        for source_path in search_paths:
            if os.path.exists(source_path):
                logger.info(f"📋 Copying CSV to import dir:")
                logger.info(f"   From: {source_path}")
                logger.info(f"   To:   {target_path}")
                shutil.copy(source_path, target_path)
                logger.info(f"✅ CSV copied successfully")
                return csv_basename
        
        # Nicht gefunden
        logger.error(f"❌ CSV file not found: {self.csv_filename}")
        logger.error(f"   Searched in:")
        for p in search_paths[:5]:
            logger.error(f"      - {p}")
        sys.exit(1)

    def run_load_csv(self):
        csv_basename = self.ensure_csv_in_import_dir()
        file_url = f"file:///{csv_basename}"
        logger.info(f"⚡ Loading CSV from {file_url}")
        
        db_name = os.environ.get("NEO4J_DB", config.NEO4J_DB)
        
        query = f"""
        LOAD CSV WITH HEADERS FROM '{file_url}' AS line
        CALL {{
            WITH line
            MERGE (c:Customer {{GPID: line.GPID}})
            SET c.firstName = line.FIRST_NAME, 
                c.lastName = line.LAST_NAME, 
                c.risk_score = toInteger(line.EMC_SCORE_FINAL), 
                c.fraud_label = toInteger(line.Fraud_incl_wo)
            FOREACH (_ IN CASE WHEN toInteger(line.Fraud_incl_wo) = 1 THEN [1] ELSE [] END | 
                SET c.is_fraudster = 1)
            FOREACH (_ IN CASE WHEN line.EMAIL IS NOT NULL AND line.EMAIL <> '' THEN [1] ELSE [] END | 
                MERGE (e:Email {{addr: line.EMAIL}}) 
                MERGE (c)-[:USES_EMAIL]->(e))
            MERGE (t:Transaction {{tx_id: line.RMS_PRUEF_ID}})
            SET t.amount = toFloat(line.WARENKORB_WERT), 
                t.date = datetime(line.PRUEFUNG_INTERN_DATUM), 
                t.is_fraud = toInteger(line.Fraud_incl_wo), 
                t.payment_method = line.PAYMENT_TYPE_USED, 
                t.channel = line.VERTRIEBSWEG
            MERGE (c)-[:MADE]->(t)
            MERGE (c)-[:PERFORMED]->(t)
            FOREACH (_ IN CASE WHEN line.TMX_DIGITAL_ID IS NOT NULL AND line.TMX_DIGITAL_ID <> '' THEN [1] ELSE [] END | 
                MERGE (d:Device {{fingerprint: line.TMX_DIGITAL_ID}}) 
                SET d.trust_score = toInteger(line.TMX_DEV_POLICY_SCORE) 
                MERGE (t)-[:USED_DEVICE]->(d) 
                MERGE (c)-[:USED_DEVICE]->(d) 
                FOREACH (__ IN CASE WHEN line.TMX_DEV_IPADDRESS IS NOT NULL AND line.TMX_DEV_IPADDRESS <> '' THEN [1] ELSE [] END | 
                    MERGE (ip_dev:IPAddress {{addr: line.TMX_DEV_IPADDRESS}}) 
                    MERGE (d)-[:PLACED_FROM]->(ip_dev)))
            FOREACH (_ IN CASE WHEN line.TMX_DEV_IPADDRESS IS NOT NULL AND line.TMX_DEV_IPADDRESS <> '' THEN [1] ELSE [] END | 
                MERGE (ip:IPAddress {{addr: line.TMX_DEV_IPADDRESS}}) 
                MERGE (t)-[:FROM_IP]->(ip))
            WITH c, line
            MERGE (a:Address {{addrHash: coalesce(line.LA_STRASSE,'') + '|' + coalesce(line.LA_PLZ,'')}})
            SET a.street = line.LA_STRASSE, 
                a.zip = line.LA_PLZ, 
                a.city = line.LA_ORT
            MERGE (c)-[:SHIPPED_TO]->(a)
        }} IN TRANSACTIONS OF 1000 ROWS
        """
        
        try:
            with self.driver.session(database=db_name) as session:
                session.run(query).consume()
            logger.info("🎉 Import Successful!")
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise e

    def run(self):
        self.connect()
        try:
            self.wait_for_database()
            self.create_constraints()
            self.run_load_csv()
        except Exception as e:
            logger.error(f"❌ Loader failed: {e}")
            sys.exit(1)
        finally:
            self.close()


if __name__ == "__main__":
    loader = FraudGraphLoader()
    loader.run()
