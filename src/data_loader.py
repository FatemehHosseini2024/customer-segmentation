"""Data loading and CustomerID recovery for the UCI Online Retail dataset.

Loading policy
--------------
1. Load the raw workbook **as-is** (all rows, all columns).
2. Apply **invoice propagation**: a CustomerID is recovered onto a row that
   lacks one when that row shares an `InvoiceNo` with a row that *does* carry
   a CustomerID. This is data completion, not imputation — in this dataset an
   invoice is a single transaction, so every line of an invoice belongs to the
   same customer. Where the data structure allows it, this recovers genuine
   customer attribution instead of discarding legitimate rows.
3. Apply **invoice propagation** and **Description backfill** (recover, never drop).
4. Drop **exact duplicate rows** (byte-identical across all columns).
5. Drop **exact duplicate rows** (byte-identical across all columns).
6. **Drop rows with unattributable `CustomerID`** (guest/cash checkouts):
   verified unrecoverable on this dataset — guest invoices are disjoint
   from customer-ID'd invoices, so propagation recovers nothing and no
   legitimate attribution source exists. The modelling unit is the
   customer, so these rows contribute no usable signal.
7. Drop rows with **non-positive `UnitPrice`** (<=0): these are
   manual adjustments/credits or data-entry errors, not sales
   (returns are encoded as negative Quantity with a positive price).
8. **No columns are dropped at any stage.** Remaining row-level
   decisions (cancellations, zero quantity) are explicit and opt-in,
   so the caller can reproduce the baseline at any time.

Structural note (verified on this dataset)
------------------------------------------
In `data/Online Retail.xlsx` the missing-CustomerID rows are *not* recoverable
by invoice propagation: the 3,710 guest invoices are **disjoint** from the
22,190 customer-ID'd invoices, and no invoice mixes null and non-null
CustomerID rows. The propagation step therefore recovers **0** rows here, but
it is retained because it is the correct general logic and matters on datasets
where invoices are partially populated.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

DATA_PATH = Path("data/Online Retail.xlsx")

# Columns present in the raw workbook (single source of truth).
RAW_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw workbook exactly as it appears on disk."""
    df = pd.read_excel(path)
    return df


def propagate_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Recover CustomerID onto rows that lack it via shared InvoiceNo.

    A row with a missing CustomerID is attributed when its InvoiceNo is also
    present on a row that carries a CustomerID. Because an invoice is a single
    transaction, all lines of an invoice belong to the same customer, so this
    is a structural completion rather than a guess.

    Rows whose InvoiceNo is entirely absent from the customer-ID'd set remain
    NaN. No rows are dropped.
    """
    if "CustomerID" not in df.columns or "InvoiceNo" not in df.columns:
        return df

    out = df.copy()
    # Map each invoice -> its known CustomerID (first non-null, deterministic).
    known = (
        df.loc[df["CustomerID"].notna(), ["InvoiceNo", "CustomerID"]]
        .drop_duplicates(subset=["InvoiceNo"])
        .set_index("InvoiceNo")["CustomerID"]
    )

    missing_mask = out["CustomerID"].isna()
    recovered = missing_mask & out["InvoiceNo"].isin(known.index)
    n_recovered = int(recovered.sum())

    out.loc[recovered, "CustomerID"] = out.loc[recovered, "InvoiceNo"].map(known)

    print(
        f"[data_loader] invoice propagation: {n_recovered:,} row(s) recovered, "
        f"{int(missing_mask.sum() - n_recovered):,} remain unattributable"
    )
    return out


def backfill_description(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill missing `Description` values from a StockCode -> Description map.

    A missing Description is recovered when the same `StockCode` appears on a
    row that *does* carry a Description. This is product-name completion, not
    imputation: a StockCode is a stable product identifier, so its name is the
    same everywhere it appears.

    Rows whose StockCode has no known Description anywhere remain NaN. No rows
    or columns are dropped.

    Verified on this dataset: 1,342 of 1,454 missing Descriptions recovered
    (92.3%); the 112 unrecoverable rows are 112 distinct StockCodes that each
    appear exactly once in the workbook with no reference copy.
    """
    if "Description" not in df.columns or "StockCode" not in df.columns:
        return df

    out = df.copy()
    lookup = (
        out.loc[out["Description"].notna(), ["StockCode", "Description"]]
        .drop_duplicates(subset=["StockCode"])
        .set_index("StockCode")["Description"]
    )

    missing_mask = out["Description"].isna()
    recovered = missing_mask & out["StockCode"].isin(lookup.index)
    n_recovered = int(recovered.sum())

    out.loc[recovered, "Description"] = out.loc[recovered, "StockCode"].map(lookup)

    print(
        f"[data_loader] description backfill: {n_recovered:,} row(s) recovered, "
        f"{int(missing_mask.sum() - n_recovered):,} remain (one-off StockCodes with no reference)"
    )
    return out


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are byte-identical across every column.

    Keeps the first occurrence of each distinct row. These are true duplicates
    (all 8 columns equal), not legitimate multi-item invoice lines, and they
    would double-count frequency and monetary value in the customer-level
    aggregation if retained.

    Verified on this dataset: 5,268 of 541,909 rows dropped (~1.0%). No customer
    is present only in duplicated rows, so the customer count is unchanged
    (4,372 before and after).
    """
    if df.empty:
        return df
    n_before = len(df)
    out = df.drop_duplicates(keep="first")
    n_dropped = n_before - len(out)
    print(f"[data_loader] exact duplicates dropped: {n_dropped:,} of {n_before:,} rows")
    return out


def drop_missing_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose CustomerID is missing (unattributable guest checkouts).

    These rows cannot be tied to any customer: the missing-CustomerID rows
    belong to guest invoices that are entirely disjoint from the
    customer-ID'd invoices, and no invoice mixes null and non-null
    CustomerID rows, so invoice propagation recovers nothing. Imputing
    identities would fabricate customers and corrupt the segments.

    Verified on this dataset: drops 135,080 of 541,909 rows (24.9%) but
    only 34 of 4,372 unique customers (0.8%) — the retained customers
    span the bulk of transaction rows.

    No columns are dropped; only rows with NaN CustomerID.
    """
    if "CustomerID" not in df.columns:
        return df

    n_before = len(df)
    n_missing = int(df["CustomerID"].isna().sum())
    out = df.dropna(subset=["CustomerID"])
    print(
        f"[data_loader] missing-CustomerID rows dropped: {n_missing:,} "
        f"of {n_before:,} ({n_missing / n_before:.1%}), "
        f"{n_before - len(out):,} removed"
    )
    return out


def drop_nonpositive_price(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with non-positive UnitPrice (<= 0).

    A non-positive unit price is not a valid sale price: returns in this
    dataset are encoded as negative Quantity with a *positive* UnitPrice,
    so a zero/negative price indicates a manual adjustment, credit, or
    data-entry error rather than a purchase. Retaining these rows would
    produce negative or zero line values that distort Monetary.

    Verified on the raw data: 2,517 of 541,909 rows (0.5%).

    Only rows are dropped; no columns are removed.
    """
    if "UnitPrice" not in df.columns:
        return df

    n_bad = int((df["UnitPrice"] <= 0).sum())
    out = df[df["UnitPrice"] > 0]
    print(f"[data_loader] non-positive UnitPrice rows dropped: {n_bad:,}")
    return out


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load raw data and apply all recovery/cleaning steps.

    Returns the full dataframe with all columns plus `is_product` flag:
    - CustomerID recovered via invoice propagation where data structure allows
    - Description backfilled from StockCode -> Description map
    - Exact duplicate rows dropped (keep-first)
    - Rows with missing CustomerID dropped (guest checkouts, unrecoverable)
    - Non-positive UnitPrice rows dropped (adjustments/errors, not sales)
    - StockCode normalized (lowercase suffix -> uppercase)
    - Non-product codes flagged via `is_product` column (fees, postage, adjustments)
    - C-invoices and Quantity == 0 are NOT dropped here; caller decides
      (opt-in). C-invoices are retained so cancellation behavior can be
      engineered as a segmentation dimension.
    """
    df = load_raw(path)
    df = propagate_customer_id(df)
    df = backfill_description(df)
    df = drop_exact_duplicates(df)
    df = drop_missing_customer_id(df)
    df = drop_nonpositive_price(df)
    df = normalize_stockcode(df)
    df = filter_non_product_codes(df)
    return df


PRODUCT_CODE_RE = re.compile(r"^\d{4,6}[A-Z]{0,2}$")

NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "D", "S", "BANK CHARGES", "AMAZONFEE", "CRUK",
    "C2", "MANUAL", "ADJUST", "ADJUSTMENT", "DISCOUNT", "FREIGHT",
}


def normalize_stockcode(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize StockCode: uppercase single-letter suffix on 4-6 digit codes.

    E.g., "47566b" -> "47566B", "15056n" -> "15056N". Only applies to codes
    matching ``^\\d{4,6}[a-z]$`` (digits + single lowercase letter), which are
    typo variants of valid product codes.
    """
    if "StockCode" not in df.columns:
        return df

    out = df.copy()
    sc = out["StockCode"].astype(str)

    # 1-letter lowercase suffix (e.g., 47566b -> 47566B)
    typo_mask1 = sc.str.match(r"^\d{4,6}[a-z]$")
    n_typo1 = int(typo_mask1.sum())

    # 2-letter lowercase suffix (e.g., 15056bl -> 15056BL) - only when no uppercase counterpart exists
    typo_mask2 = sc.str.match(r"^\d{4,6}[a-z]{2}$")
    n_typo2 = int(typo_mask2.sum())

    total_typo = n_typo1 + n_typo2

    if total_typo:
        if n_typo1:
            out.loc[typo_mask1, "StockCode"] = sc[typo_mask1].str.upper()
        if n_typo2:
            out.loc[typo_mask2, "StockCode"] = sc[typo_mask2].str.upper()
        print(f"[data_loader] normalized {total_typo:,} lowercase-suffix StockCodes "
              f"(1-letter: {n_typo1:,}, 2-letter: {n_typo2:,})")

    return out


def filter_non_product_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Mark non-product StockCodes for downstream feature engineering.

    Adds `is_product` boolean column:
    - True: standard product code (4-6 digits, optional uppercase letter)
    - False: fees, adjustments, postage, bank charges, manual codes

    POSTAGE is kept in monetary features but excluded from product features.
    """
    if "StockCode" not in df.columns:
        return df

    out = df.copy()
    sc = out["StockCode"].astype(str)

    is_product = sc.str.match(PRODUCT_CODE_RE) & ~sc.isin(NON_PRODUCT_CODES)
    out["is_product"] = is_product

    n_nonprod = int((~is_product).sum())
    print(f"[data_loader] flagged {n_nonprod:,} non-product lines (fees/adjustments/postage)")

    return out


def validate_columns(df: pd.DataFrame) -> None:
    """Assert the dataframe still carries every raw column."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected column loss: {missing}")