import pandas as pd
import numpy as np
import random
import uuid
import os
import config
from faker import Faker

# Initialize Faker
fake = Faker('de_DE')
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ==========================================
# CONFIGURATION
# ==========================================
NUM_ROWS = 25000
FRAUD_RATE = 0.05
NUM_FRAUDSTERS = int(NUM_ROWS * FRAUD_RATE)
NUM_LEGIT = NUM_ROWS - NUM_FRAUDSTERS

# Graph Topologie (Realistic "Messy" Signals)
NUM_FRAUD_RINGS = 50  # Many small rings
SHARED_DEVICE_PROB = 0.50  # Fraudsters only share devices 50% of the time (Harder for Graph)
SHARED_IP_PROB = 0.40  # IPs are shared even less often
SUPERNODE_IPS = [fake.ipv4() for _ in range(5)]  # Legit users share these IPs
SUPERNODE_PROB = 0.15  # 15% of legit traffic is on "University/Corp" IPs

# ==========================================
# 1. SETUP POOLS
# ==========================================
fraud_rings = []
for _ in range(NUM_FRAUD_RINGS):
    fraud_rings.append({
        'device_pool': [uuid.uuid4().hex for _ in range(random.randint(2, 5))],
        'ip_pool': [fake.ipv4() for _ in range(2)],
        'email_domain': fake.free_email_domain(),
        'drop_address': {
            'street': fake.street_name(),
            'number': fake.building_number(),
            'zip': fake.postcode(),
            'city': fake.city()
        }
    })


# ==========================================
# 2. GENERATOR FUNKTION
# ==========================================
def generate_row(is_fraud=False):
    tx_id = f"ORD{uuid.uuid4().hex[:12].upper()}"
    gpid = f"GP{random.randint(100000, 999999)}"

    # --- REALISTIC BASELINE: Overlap but distinct ---
    # Fraudsters generally spend more, but legit users also buy expensive things.
    if is_fraud:
        # Fraud: Log-Normal (Mean ~100€, but long tail to 1000€)
        basket_value = np.random.lognormal(mean=4.6, sigma=0.9)
    else:
        # Legit: Log-Normal (Mean ~90€)
        basket_value = np.random.lognormal(mean=4.5, sigma=0.8)

    basket_value = round(basket_value, 2)

    # Payment Types: Fraudsters prefer Credit Cards slightly more, but heavily mixed
    if is_fraud:
        payment_type = np.random.choice(['Credit Card', 'PayPal', 'Invoice'], p=[0.5, 0.3, 0.2])
    else:
        payment_type = np.random.choice(['Credit Card', 'PayPal', 'Invoice'], p=[0.3, 0.4, 0.3])

    # PII
    gender = np.random.choice(['M', 'F'])
    first_name = fake.first_name_male() if gender == 'M' else fake.first_name_female()
    last_name = fake.last_name()

    legit_street = fake.street_name()
    legit_num = fake.building_number()
    legit_zip = fake.postcode()
    legit_city = fake.city()

    # --- GRAPH LOGIC (The Only Clear Signal) ---
    if is_fraud:
        ring = random.choice(fraud_rings)
        # Identity Morphing (Graph Signal)
        email = f"{first_name.lower()}{random.randint(10, 99)}@{ring['email_domain']}"

        # Device/IP Linkage (The Graph Signal)
        digital_id = random.choice(ring['device_pool']) if random.random() < SHARED_DEVICE_PROB else uuid.uuid4().hex
        ip_address = random.choice(ring['ip_pool']) if random.random() < SHARED_IP_PROB else fake.ipv4()

        # Address Sharing
        ra_strasse = ring['drop_address']['street']
        ra_hausnr = ring['drop_address']['number']
        ra_plz = ring['drop_address']['zip']
        ra_ort = ring['drop_address']['city']

        # RISK SCORES: "Decent" Separation
        # Fraud Mean: 550, Legit Mean: 250.
        # This gives XGBoost a chance (AUC ~0.60-0.70), but not a free win.
        policy_score = int(np.random.normal(550, 150))

    else:
        email = f"{first_name.lower()}.{last_name.lower()}@example.{'de' if random.random() > 0.2 else 'com'}"
        digital_id = uuid.uuid4().hex

        # Supernode Logic (Graph Confusion)
        if random.random() < SUPERNODE_PROB:
            ip_address = random.choice(SUPERNODE_IPS)
        else:
            ip_address = fake.ipv4()

        ra_strasse = legit_street
        ra_hausnr = legit_num
        ra_plz = legit_zip
        ra_ort = legit_city

        # RISK SCORES
        policy_score = int(np.random.normal(250, 150))

    policy_score = np.clip(policy_score, 0, 1000)

    # REMOVE LEAKAGE: Reason codes are random noise
    reason_code = np.random.choice(["OK", "VERIFY", "RISK"], p=[0.8, 0.15, 0.05])

    return {
        "RMS_PRUEF_ID": tx_id,
        "GPID": gpid,
        "PRUEFUNG_INTERN_DATUM": fake.date_time_this_year(),
        "WARENKORB_WERT": basket_value,
        "PAYMENT_TYPE_USED": payment_type,
        "VERTRIEBSWEG": "Web",
        "Fraud_incl_wo": 1 if is_fraud else 0,
        "EMAIL": email,
        "FIRST_NAME": first_name,
        "LAST_NAME": last_name,
        "KUNDENKLASSE": "Neukunde",
        "KUNDENGRUPPE": 1,
        "FLAG_ERSTBESTELLER": 0,
        "RA_STRASSE": ra_strasse,
        "LA_STRASSE": ra_strasse,
        "RA_HAUSNUMMER": ra_hausnr,
        "LA_HAUSNUMMER": ra_hausnr,
        "RA_PLZ": ra_plz,
        "LA_PLZ": ra_plz,
        "RA_ORT": ra_ort,
        "LA_ORT": ra_ort,
        "FLAG_ABW_LIEFERUNG": 0,
        "TMX_DIGITAL_ID": digital_id,
        "TMX_DEV_IPADDRESS": ip_address,
        "TMX_DEV_POLICY_SCORE": policy_score,
        "TMX_DEV_CITY": fake.city(),
        "TMX_DEV_COUNTRY": "DE",
        "EMC_SCORE_FINAL": np.clip(policy_score / 10, 0, 100),
        "IDS_VALUE": "A",
        "IDS_TRUST_RANK_VALUE": 5,
        "TMX_DEV_REGION": fake.state(),
        "SESSION_ID": f"SES{uuid.uuid4().hex[:10]}",
        "RMS_PRUEF_ID_MASTER": f"MAS{uuid.uuid4().hex[:10]}",
        "ERGEBNIS_EXTERN_SCORE": 500,
        "AUSKUNFTEILIMIT": 1000,
        "BP_ORDERS_NSZA_12M_OWN": 0,
        "BP_ORDERS_NSZA_12M_DUPLICATES": 0,
        "DUBLETTEN_ANZ_TREFFER": 0,
        "BEZEICHNUNG_AUSKUNFTEI": "Schufa",
        "PERSONENTREFFERART_EXTERN": "None",
        "CustomerGroupZoot": 0,
        "TMX_DEV_DEVICE_REASON_CODE": reason_code,
        "TMX_DEV_SUMMARY_REASON_CODE": reason_code,
        "fold": 1
    }


# ==========================================
# 3. EXECUTION & SAVING
# ==========================================
print(f"Generating {NUM_ROWS} rows (REALISTIC MODE)...")
data = []
for _ in range(NUM_FRAUDSTERS): data.append(generate_row(is_fraud=True))
for _ in range(NUM_LEGIT): data.append(generate_row(is_fraud=False))

random.shuffle(data)
df = pd.DataFrame(data)

# Save Files
local_filename = "transactions.csv"
df.to_csv(local_filename, index=False)
print(f"Saved local copy: {local_filename}")

# Optional: Save to Neo4j Import folder if configured
if hasattr(config, 'NEO4J_IMPORT_DIR') and config.NEO4J_IMPORT_DIR:
    try:
        neo4j_path = os.path.join(config.NEO4J_IMPORT_DIR, config.CSV_FILENAME)
        df.to_csv(neo4j_path, index=False)
        print(f"Saved Neo4j copy: {neo4j_path}")
    except Exception as e:
        print(f"Skipping Neo4j import folder save: {e}")

print(f"Dataset Stats:")
print(f"- Rows: {len(df)}")
print(f"- Fraud Rate: {df['Fraud_incl_wo'].mean():.2%}")
print(f"- Avg Fraud Score: {df[df['Fraud_incl_wo'] == 1]['TMX_DEV_POLICY_SCORE'].mean():.1f}")
print(f"- Avg Legit Score: {df[df['Fraud_incl_wo'] == 0]['TMX_DEV_POLICY_SCORE'].mean():.1f}")