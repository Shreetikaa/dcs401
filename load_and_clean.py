"""
load_and_clean.py
------------------
Just the "load and clean" step of the pipeline, split out on its own.

Loads the raw "Online Retail" dataset (data/online_retail.csv) and applies
the same cleaning rules used in prepare_db.py, but WITHOUT touching SQLite:
  - drops rows with no Description
  - drops non-product housekeeping codes (POSTAGE, DOTCOM POSTAGE, etc.)
  - drops rows whose Description is actually a staff inventory-adjustment
    note rather than a real product (e.g. "damaged", "check", "thrown away")
  - flags cancelled orders (InvoiceNo starting with 'C') as returns
  - computes line_total = Quantity * UnitPrice
  - derives invoice_date, year_month, weekday, hour

Saves the cleaned result to data/online_retail_cleaned.csv so it can be
loaded directly by any other script (pandas.read_csv), without needing to
touch the raw file or the SQLite database again.

Run with:  python load_and_clean.py
"""

import pandas as pd

CSV_PATH = "data/online_retail.csv"
CLEANED_CSV_PATH = "data/online_retail_cleaned.csv"

print("Loading raw CSV...")
df = pd.read_csv(CSV_PATH, encoding="latin1")
print(f"  raw rows: {len(df):,}")

# --- rename columns to clean snake_case ----------------------------------
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={
    "InvoiceNo": "invoice_no", "StockCode": "stock_code", "Description": "description",
    "Quantity": "quantity", "InvoiceDate": "invoice_date", "UnitPrice": "unit_price",
    "CustomerID": "customer_id", "Country": "country",
})

# --- drop non-product housekeeping codes ---------------------------------
JUNK_CODES = {"POST", "DOT", "D", "M", "BANK CHARGES", "PADS", "CRUK", "C2"}
df["stock_code"] = df["stock_code"].astype(str).str.upper()
n_before = len(df)
df = df[~df["stock_code"].isin(JUNK_CODES)]
print(f"  dropping {n_before - len(df):,} non-product housekeeping rows (postage, fees, etc.)")

# --- flag cancelled orders as returns (kept, not dropped) ----------------
df["invoice_no"] = df["invoice_no"].astype(str)
df["is_return"] = df["invoice_no"].str.startswith("C")

# --- drop rows with no product description -------------------------------
df["description"] = df["description"].str.strip()
n_before = len(df)
df = df.dropna(subset=["description"])
print(f"  dropping {n_before - len(df):,} rows with no product description")

# --- drop staff inventory-adjustment notes masquerading as products ------
# exact-match list (not a substring match, since words like "check" also
# appear inside real product names such as "SUNSET CHECK HAMMOCK")
JUNK_DESCRIPTIONS = {
    "check", "samples", "?", "??", "???", "damages", "damages?", "found",
    "adjustment", "thrown away", "missing", "smashed", "test", "wet damaged",
    "water damaged", "damaged stock", "samples/damages",
}
desc_lower = df["description"].str.lower()
is_junk_desc = desc_lower.isin(JUNK_DESCRIPTIONS) | desc_lower.str.startswith("wrongly")
print(f"  dropping {int(is_junk_desc.sum())} inventory-adjustment-note rows (not real products)")
df = df[~is_junk_desc]

# --- derive date-based columns --------------------------------------------
df["invoice_date"] = pd.to_datetime(df["invoice_date"])
df["invoice_date_str"] = df["invoice_date"].dt.date.astype(str)
df["year_month"] = df["invoice_date"].dt.to_period("M").astype(str)
df["weekday"] = df["invoice_date"].dt.day_name()
df["hour"] = df["invoice_date"].dt.hour

# --- compute revenue per line item ----------------------------------------
df["line_total"] = df["quantity"] * df["unit_price"]

# --- summary ---------------------------------------------------------------
print(f"\nCleaned rows: {len(df):,}")
print(f"  rows with a known customer: {df['customer_id'].notna().sum():,}")
print(f"  unique customers: {df['customer_id'].nunique():,}")
print(f"  unique products: {df['stock_code'].nunique():,}")
print(f"  returns (cancelled invoices): {int(df['is_return'].sum()):,} line items")
print(f"  date range: {df['invoice_date_str'].min()} to {df['invoice_date_str'].max()}")

# --- save the cleaned data --------------------------------------------------
df.to_csv(CLEANED_CSV_PATH, index=False)
print(f"\nDone. Cleaned data saved to {CLEANED_CSV_PATH}")
