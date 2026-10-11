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
5. **No columns are dropped at any stage.** Every remaining row-level
   cleaning decision (unattributable CustomerID, cancellations, zero price,
   zero quantity) is explicit and opt-in, so the caller can reproduce the raw
   baseline at any time.

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


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load raw data and apply all three recovery/cleaning steps.

    Returns the full dataframe with all columns: CustomerID is recovered via
    invoice propagation where the data structure allows it; Description is
    backfilled from the StockCode -> Description map; exact duplicate rows are
    dropped. Rows that remain unattributable are kept (with NaN CustomerID) so
    the caller can decide how to handle them. No columns are dropped.
    """
    df = load_raw(path)
    df = propagate_customer_id(df)
    df = backfill_description(df)
    df = drop_exact_duplicates(df)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Assert the dataframe still carries every raw column."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected column loss: {missing}")