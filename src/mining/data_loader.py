"""Load, clean and reshape the Online Retail transactions.

Output of `clean_transactions` is one row per (basket, item) with:
    basket_id, item, quantity, unit_price, utility, invoice_date,
    weekday, is_weekend, month, hour, country
`utility` = quantity x unit_price  (what high-utility mining needs).
"""
from pathlib import Path
import pandas as pd

from src.core import config as C


# ---------- loading ----------
def find_raw_file() -> Path:
    for p in C.RAW_FILE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Raw dataset not found. Download UCI 'Online Retail' and save it as "
        f"{C.RAW_FILE_CANDIDATES[0]}"
    )


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else find_raw_file()
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="latin-1")


# ---------- cleaning ----------
def _drop_cancelled_originals(df: pd.DataFrame) -> pd.DataFrame:
    """Remove purchase lines that were later cancelled in full.

    A cancellation is an invoice starting with 'C' with a negative quantity.
    If the same customer cancelled the same item with exactly the opposite
    quantity on/after the purchase date, the original line is dropped too.
    (Otherwise one cancelled bulk order can dominate utility results.)
    """
    if C.COL_CUSTOMER not in df.columns:
        return df
    is_cancel = df[C.COL_INVOICE].str.startswith("C")
    cancels = df[is_cancel & df[C.COL_CUSTOMER].notna()].copy()
    buys = df[~is_cancel & df[C.COL_CUSTOMER].notna()].copy()
    if cancels.empty or buys.empty:
        return df

    key = [C.COL_CUSTOMER, C.COL_STOCK]
    cancels["match_qty"] = -cancels[C.COL_QTY]
    cancels = cancels[cancels["match_qty"] > 0]
    cancels[C.COL_DATE] = pd.to_datetime(cancels[C.COL_DATE])
    buys[C.COL_DATE] = pd.to_datetime(buys[C.COL_DATE])

    merged = buys.reset_index().merge(
        cancels[key + ["match_qty", C.COL_DATE]].rename(columns={C.COL_DATE: "cancel_date"}),
        left_on=key + [C.COL_QTY], right_on=key + ["match_qty"], how="inner",
    )
    merged = merged[merged["cancel_date"] >= merged[C.COL_DATE]]
    return df.drop(index=merged["index"].unique())

def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # 1. drop rows missing what we need
    df = df.dropna(subset=[C.COL_INVOICE, C.COL_STOCK, C.COL_DESC, C.COL_DATE])

    # 2. normalise types / text
    df[C.COL_INVOICE] = df[C.COL_INVOICE].astype(str).str.strip()
    df[C.COL_STOCK] = df[C.COL_STOCK].astype(str).str.strip().str.upper()
    df[C.COL_DESC] = df[C.COL_DESC].astype(str).str.strip().str.upper()

    # 3. net out fully-cancelled purchases, then drop cancellations, bad rows, fees
    df = _drop_cancelled_originals(df)
    df = df[~df[C.COL_INVOICE].str.startswith("C")]
    df = df[(df[C.COL_QTY] > 0) & (df[C.COL_PRICE] > 0)]
    df = df[~df[C.COL_STOCK].isin(C.NON_PRODUCT_CODES)]

    # 4. one canonical name per StockCode (descriptions vary for the same code)
    canonical = (
        df.groupby(C.COL_STOCK)[C.COL_DESC]
        .agg(lambda s: s.value_counts().idxmax())
    )
    df["item"] = df[C.COL_STOCK].map(canonical)

    # 5. utility per line, then collapse to one row per (basket, item)
    df["utility"] = df[C.COL_QTY] * df[C.COL_PRICE]
    df[C.COL_DATE] = pd.to_datetime(df[C.COL_DATE])

    out = (
        df.groupby([C.COL_INVOICE, "item"], as_index=False)
        .agg(
            quantity=(C.COL_QTY, "sum"),
            unit_price=(C.COL_PRICE, "mean"),
            utility=("utility", "sum"),
            invoice_date=(C.COL_DATE, "first"),
            country=(C.COL_COUNTRY, "first"),
        )
        .rename(columns={C.COL_INVOICE: "basket_id"})
    )

    # 6. time features (used later for filters like "weekends")
    out["weekday"] = out["invoice_date"].dt.day_name()
    out["is_weekend"] = out["invoice_date"].dt.day_of_week >= 5
    out["month"] = out["invoice_date"].dt.month
    out["hour"] = out["invoice_date"].dt.hour
    return out


# ---------- save / load processed ----------
def prepare_and_save(raw_path: Path | None = None) -> pd.DataFrame:
    clean = clean_transactions(load_raw(raw_path))
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    clean.to_csv(C.CLEAN_FILE, index=False)
    return clean


def load_clean() -> pd.DataFrame:
    if not C.CLEAN_FILE.exists():
        raise FileNotFoundError("Run `python -m src.prepare_data` first.")
    return pd.read_csv(C.CLEAN_FILE, parse_dates=["invoice_date"])


# ---------- shapes the mining algorithms need ----------
def to_baskets(df: pd.DataFrame) -> list[list[str]]:
    """List of baskets, each a list of item names (for Apriori/FP-Growth)."""
    return df.groupby("basket_id")["item"].apply(list).tolist()


def summarize(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "baskets": df["basket_id"].nunique(),
        "distinct_items": df["item"].nunique(),
        "total_utility": round(float(df["utility"].sum()), 2),
        "date_range": (str(df["invoice_date"].min()), str(df["invoice_date"].max())),
    }
