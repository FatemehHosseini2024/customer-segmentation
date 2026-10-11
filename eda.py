"""
Initial Exploratory Data Analysis for the UCI Online Retail dataset.

Purpose
    Give a first, reproducible look at the RAW transaction data BEFORE the
    cleaning pipeline is applied: shape, dtypes, missing values, duplicates,
    invoice cancellations, quantity/unit-price anomalies, non-product line
    items, cardinality, the date span, geography, a monetary preview, and a
    preview of how the cleaning policy affects row counts.

Run
    python eda.py

Inputs
    data/Online Retail.xlsx

Outputs
    reports/eda_report.md        human-readable markdown report
    reports/eda_missing.csv      per-column missing value counts
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/Online Retail.xlsx")
REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "eda_report.md"
MISSING_CSV = REPORT_DIR / "eda_missing.csv"


def df_to_md(frame: pd.DataFrame, floatfmt: str = "%.3f") -> str:
    cols = [str(c) for c in frame.columns]

    def fmt(v) -> str:
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            if pd.isna(v):
                return ""
            return floatfmt % v
        if isinstance(v, (pd.Timestamp,)):
            return str(v)
        if pd.isna(v):
            return ""
        return str(v)

    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def build_report(df: pd.DataFrame) -> str:
    n = len(df)
    md: list[str] = []
    md.append("# Initial EDA — UCI Online Retail (raw transactions)\n")
    md.append(f"> Source: `{DATA_PATH}`\n")
    md.append(f"> Tool: `eda.py`. **{n:,} rows x {df.shape[1]} columns.**\n")

    # 1. Structure
    md.append("## 1. Structure\n")
    md.append("| Column | dtype | non-null |")
    md.append("|---|---|---|")
    for c in df.columns:
        md.append(f"\n| {c} | {df[c].dtype} | {df[c].notna().sum():,} |")

    # 2. Missing values
    md.append("\n## 2. Missing values\n")
    miss = df.isna().sum()
    miss_pct = (miss / n * 100).round(2)
    miss_df = pd.DataFrame({"count": miss.values, "pct": miss_pct.values}, index=df.columns)
    miss_df.to_csv(MISSING_CSV, index_label="column")
    md.append(f"Full per-column table also written to `{MISSING_CSV}`.\n")
    md.append(df_to_md(miss_df.reset_index().rename(columns={"index": "column", "pct": "pct(%)"})).replace("pct(%)", "pct (%)") + "\n")
    md.append("Key observations:\n")
    md.append(f"- `CustomerID` is missing in **{miss['CustomerID']:,}** rows (~{miss['CustomerID']/n:.1%}). These are non-registered / cash-checkout orders that cannot be tied to a customer; they are dropped before feature engineering.")
    md.append(f"- `Description` is missing in **{miss['Description']:,}** rows (packing/discount lines).")
    md.append(f"- `InvoiceDate` is missing in **{miss['InvoiceDate']:,}** rows.\n" if miss['InvoiceDate'] else "- `InvoiceDate` has no missing values.\n")
    md.append("All other columns are fully populated (InvoiceNo, StockCode, Quantity, UnitPrice, Country).")

    # 3. Duplicates
    md.append("\n## 3. Duplicates\n")
    full_dups = df.duplicated().sum()
    key_cols = ["InvoiceNo", "StockCode", "CustomerID", "InvoiceDate"]
    key_dups = df.duplicated(subset=key_cols, keep=False).sum()
    md.append(f"- Exact duplicate rows: **{full_dups:,}** — removed in the cleaning step.")
    md.append(f"- Rows sharing (InvoiceNo, StockCode, CustomerID, InvoiceDate): **{key_dups:,}**. Many are legitimate multi-item invoice lines, but the exact full-row duplicates above are the ones dropped.")

    # 4. Cancellations
    inv = df["InvoiceNo"].astype(str)
    canc = inv.str.startswith("C")
    md.append("\n## 4. Invoice cancellations\n")
    md.append(f"- Invoices prefixed with `C` (order cancellations): **{canc.sum():,}** rows (~{canc.sum()/n:.1%}).")
    md.append("- Per the cleaning policy these cancellation invoices are dropped (they represent voided orders, not purchasing behaviour).")

    # 5. Quantity
    md.append("\n## 5. Quantity\n")
    md.append(df_to_md(df["Quantity"].describe().round(1).to_frame().T))
    md.append(f"\n- Negative quantities (returns): **{(df['Quantity'] < 0).sum():,}** rows → retained and netted into Monetary value.")
    md.append(f"- Zero quantities: **{(df['Quantity'] == 0).sum():,}** rows → dropped.")
    md.append(f"- Largest order: {int(df['Quantity'].max()):,} units (a bulk / wholesale order; treated as valid).")

    # 6. UnitPrice
    md.append("\n## 6. UnitPrice\n")
    md.append(df_to_md(df["UnitPrice"].describe().round(2).to_frame().T))
    md.append(f"\n- Non-positive unit prices (<=0): **{(df['UnitPrice'] <= 0).sum():,}** rows → dropped.")

    # 7. Non-product line items
    md.append("\n## 7. Non-product line items\n")
    sc = df["StockCode"].astype(str)
    product_re = r"^\d{4,6}[A-Z]?$"
    nonprod = sc[~sc.str.match(product_re)]
    vc = nonprod.value_counts().head(12)
    tbl = pd.DataFrame({"StockCode": vc.index, "count": vc.values})
    md.append(f"StockCodes that are not standard numeric product codes (postage, bank charges, DOT adjustments, manuals, etc.:\n")
    md.append(df_to_md(tbl))
    md.append(f"\n- Total non-product lines: **{nonprod.shape[0]:,}** (~{nonprod.shape[0]/n:.1%}).")
    md.append("- These do not represent distinct products and would inflate `distinct_products`; the feature layer defines products via `StockCode` generally, so this is noted rather than filtered here.")

    # 8. Cardinality
    md.append("\n## 8. Cardinality / coverage\n")
    card = pd.DataFrame({
        "dimension": ["Unique InvoiceNo", "Unique StockCode", "Unique Description",
                      "Unique Country", "Unique CustomerID (non-null)"],
        "count": [df["InvoiceNo"].nunique(), df["StockCode"].nunique(),
                  df["Description"].nunique(), df["Country"].nunique(),
                  df["CustomerID"].nunique()],
    })
    md.append(df_to_md(card))

    # 9. InvoiceDate range + monthly activity
    md.append("\n## 9. InvoiceDate range & monthly activity\n")
    md.append(f"- Min: `{df['InvoiceDate'].min()}` to Max: `{df['InvoiceDate'].max()}` (~{(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} days).")
    monthly = df.dropna(subset=["InvoiceDate"]).set_index("InvoiceDate").resample("ME").size()
    mt = monthly.rename("rows").to_frame().reset_index()
    mt.columns = ["month_end", "rows"]
    mt["month_end"] = mt["month_end"].dt.strftime("%Y-%m")
    md.append("\nTransactions (rows) per month:\n")
    md.append(df_to_md(mt))
    md.append("- The data ends mid-December 2011 (truncated final month). The snapshot/reference date used for Recency is `max(InvoiceDate)`.")

    # 10. Geography
    md.append("\n## 10. Geography (Country)\n")
    cc = df["Country"].value_counts()
    geo = pd.DataFrame({"country": cc.index, "rows": cc.values, "share_pct": (cc.values / n * 100).round(2)})
    md.append(df_to_md(geo.head(10)))
    md.append(f"\nUnited Kingdom dominates (~{cc.iloc[0]/n:.1%}). With 38 countries but >90% UK, country is a near-constant feature and is excluded from the customer-level feature set (low signal at customer granularity).")

    # 11. Monetary preview
    md.append("\n## 11. Monetary preview (Quantity x UnitPrice)\n")
    line_value = df["Quantity"] * df["UnitPrice"]
    md.append(df_to_md(line_value.describe().round(2).to_frame().T))
    md.append("\nNegative values (returns) and the long positive tail are expected; netting per customer is applied in feature engineering.")

    # 12. Cleaning impact preview
    md.append("\n## 12. Cleaning policy impact (preview)\n")
    clean = df.dropna(subset=["CustomerID"]).copy()
    clean["InvoiceNo"] = clean["InvoiceNo"].astype(str)
    clean = clean[~clean["InvoiceNo"].str.startswith("C")]
    clean = clean[clean["UnitPrice"] > 0]
    clean = clean[clean["Quantity"] != 0]
    md.append("Cleaning policy (as implemented): drop missing `CustomerID`, drop `C`-prefix (cancellation) invoices, drop `UnitPrice <= 0`, drop `Quantity == 0`; retain & net returns.")
    md.append("| Metric | Raw | After cleaning |")
    md.append("|---|---|---|")
    md.append(f"\n| Rows | {n:,} | {len(clean):,} ({len(clean)/n:.1%}) |")
    md.append(f"| Unique customers | {df['CustomerID'].nunique():,} | {clean['CustomerID'].nunique():,} |")
    md.append("\nPer-category drop counts (note: categories overlap, so their sum exceeds the total rows dropped):\n")
    md.append("| Category dropped | Rows |")
    md.append("|---|---|")
    md.append(f"\n| Missing CustomerID | {int(miss['CustomerID']):,} |")
    md.append(f"| C-prefix (cancellation) invoices | {int(canc.sum()):,} |")
    md.append(f"| UnitPrice <= 0 | {int((df['UnitPrice'] <= 0).sum()):,} |")
    md.append(f"| Quantity == 0 | {int((df['Quantity'] == 0).sum()):,} |")

    return "\n".join(md) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading", DATA_PATH, "...")
    df = pd.read_excel(DATA_PATH)
    print("Loaded:", df.shape)
    md = build_report(df)
    REPORT_PATH.write_text(md, encoding="utf-8")
    print("Wrote report ->", REPORT_PATH)
    print("Wrote missing-values CSV ->", MISSING_CSV)
    n = len(df)
    print("\n--- EDA summary ---")
    print(f"rows={n:,} cols={df.shape[1]}")
    print(f"missing CustomerID={int(df['CustomerID'].isna().sum()):,} ({df['CustomerID'].isna().mean():.1%})")
    print(f"exact duplicate rows={int(df.duplicated().sum()):,}")
    print(f"cancellations (C-prefix)={int(df['InvoiceNo'].astype(str).str.startswith('C').sum()):,}")
    print(f"countries={df['Country'].nunique()} ; customers(non-null)={df['CustomerID'].nunique()}")
    print(f"date range={df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")


if __name__ == "__main__":
    main()
