"""Central configuration: paths and constants used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# Put the UCI "Online Retail" file here (.xlsx or .csv)
RAW_FILE_CANDIDATES = [
    DATA_RAW / "online_retail.xlsx",
    DATA_RAW / "Online Retail.xlsx",
    DATA_RAW / "online_retail.csv",
    DATA_RAW / "OnlineRetail.csv",
]

CLEAN_FILE = DATA_PROCESSED / "clean_transactions.csv"

# Raw column names in the UCI Online Retail dataset
COL_INVOICE = "InvoiceNo"
COL_STOCK = "StockCode"
COL_DESC = "Description"
COL_QTY = "Quantity"
COL_DATE = "InvoiceDate"
COL_PRICE = "UnitPrice"
COL_COUNTRY = "Country"
COL_CUSTOMER = "CustomerID"

# StockCodes that are fees/adjustments, not real products
NON_PRODUCT_CODES = {
    "POST", "D", "M", "BANK CHARGES", "DOT", "CRUK", "C2",
    "AMAZONFEE", "S", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002",
}

# Safety floor for frequent mining on Online Retail: below ~0.5% support the
# number of itemsets explodes (0.3% did not finish in 3+ minutes in testing).
MIN_SUPPORT_FLOOR = 0.005

# Rare mining safety limits (see src/mining/rare.py)
MAX_RARE_RESULTS = 200_000        # refuse settings that would return more itemsets than this
MAX_TRIPLE_CANDIDATES = 2_000_000  # refuse settings that would need more triple checks than this

# High-utility mining (SPMF, Java) settings -- see src/mining/huim.py
SPMF_JAR = ROOT / "tools" / "spmf.jar"
HUIM_TIMEOUT_S = 120            # kill the Java process if it runs longer than this
MAX_HUIM_OUTPUT_MB = 100        # kill it if its output file grows beyond this
# Lowest allowed min_utility, as a fraction of TOTAL sales value, by max itemset length.
# Measured on Online Retail: pairs at 0.02% take ~30 s, triples at 0.07% take ~45 s.
HUIM_MIN_UTILITY_PCT_FLOOR = {2: 0.0002, 3: 0.0007}

# Language layer (Step 8)
GEMINI_MODEL_DEFAULT = "gemini-3.8-flash"   # override with GEMINI_MODEL in .env (names change quickly)
LLM_TIMEOUT_S = 120
MAX_QUESTION_CHARS = 1000
LLM_LOG_FILE = ROOT / "logs" / "llm_requests.jsonl"   # every question and outcome, kept locally

# API (Step 11)
SESSION_TTL_S = 2 * 60 * 60          # an idle conversation is forgotten after 2 hours
MAX_SESSIONS = 200                   # oldest idle conversation is dropped beyond this
MAX_CONCURRENT_MINING = 2            # mining jobs running at the same time
MINING_WAIT_S = 20                   # how long a request waits for a free mining slot
RATE_LIMIT_PER_MIN = 20              # model-using requests per minute per client
CORS_ORIGINS_DEFAULT = ["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:3000"]
