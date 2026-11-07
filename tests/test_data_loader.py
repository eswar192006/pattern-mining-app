import pandas as pd
from src.mining.data_loader import clean_transactions, to_baskets, summarize


def make_raw():
    rows = [
        # InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
        ("1001", "A1", "Red Mug ", 2, "2011-01-03 10:00:00", 3.0, 1, "UK"),   # good (Mon)
        ("1001", "B2", "Blue Plate", 1, "2011-01-03 10:00:00", 5.0, 1, "UK"),  # good
        ("1001", "A1", "RED MUG", 1, "2011-01-03 10:00:00", 3.0, 1, "UK"),     # duplicate item in basket -> merged
        ("1002", "A1", "RED MUG", 4, "2011-01-08 14:30:00", 3.0, 2, "UK"),     # Saturday
        ("1002", "C3", "Green Bowl", 1, "2011-01-08 14:30:00", 7.0, 2, "UK"),
        ("C1003", "A1", "RED MUG", -2, "2011-01-09 09:00:00", 3.0, 3, "UK"),   # cancelled
        ("1004", "A1", "RED MUG", -1, "2011-01-10 09:00:00", 3.0, 3, "UK"),    # negative qty
        ("1005", "B2", "BLUE PLATE", 1, "2011-01-10 09:00:00", 0.0, 3, "UK"),  # zero price
        ("1006", "POST", "POSTAGE", 1, "2011-01-10 09:00:00", 18.0, 3, "UK"),  # non-product
        ("1007", "B2", None, 1, "2011-01-10 09:00:00", 5.0, 3, "UK"),          # missing description
    ]
    return pd.DataFrame(rows, columns=[
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country"])


def test_cleaning_rules():
    df = clean_transactions(make_raw())
    assert set(df["basket_id"]) == {"1001", "1002"}          # bad invoices removed
    assert "POSTAGE" not in set(df["item"])                   # fees removed
    assert (df["utility"] > 0).all()


def test_duplicate_items_merged_and_utility():
    df = clean_transactions(make_raw())
    mug = df[(df.basket_id == "1001") & (df.item == "RED MUG")].iloc[0]
    assert mug.quantity == 3                                  # 2 + 1
    assert mug.utility == 9.0                                 # 3 x 3.0
    assert len(df[df.basket_id == "1001"]) == 2               # mug + plate


def test_time_features():
    df = clean_transactions(make_raw())
    assert df[df.basket_id == "1001"]["is_weekend"].iloc[0] == False
    assert df[df.basket_id == "1002"]["is_weekend"].iloc[0] == True
    assert df[df.basket_id == "1002"]["weekday"].iloc[0] == "Saturday"


def test_baskets_and_summary():
    df = clean_transactions(make_raw())
    baskets = to_baskets(df)
    assert sorted(len(b) for b in baskets) == [2, 2]
    s = summarize(df)
    assert s["baskets"] == 2 and s["distinct_items"] == 3


def test_fully_cancelled_purchase_is_removed():
    raw = make_raw()
    extra = pd.DataFrame([
        # big order, then exact cancellation by the same customer -> both gone
        ("2001", "Z9", "GIANT ORDER", 5000, "2011-02-01 09:00:00", 1.0, 77, "UK"),
        ("2001", "A1", "RED MUG", 1, "2011-02-01 09:00:00", 3.0, 77, "UK"),
        ("C2002", "Z9", "GIANT ORDER", -5000, "2011-02-01 15:00:00", 1.0, 77, "UK"),
        # partial cancellation (different qty) -> original kept
        ("2003", "B2", "BLUE PLATE", 10, "2011-02-02 09:00:00", 5.0, 78, "UK"),
        ("C2004", "B2", "BLUE PLATE", -3, "2011-02-02 12:00:00", 5.0, 78, "UK"),
    ], columns=raw.columns)
    df = clean_transactions(pd.concat([raw, extra], ignore_index=True))
    assert "GIANT ORDER" not in set(df["item"])
    assert "2003" in set(df["basket_id"])
    assert "2001" in set(df["basket_id"])   # rest of that basket survives
