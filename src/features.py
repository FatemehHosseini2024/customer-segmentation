"""Customer-level feature engineering for segmentation.

Builds one row per customer from the cleaned transaction data:

RFM core (from non-cancellation invoices only):
    recency, frequency, monetary, avg_order_value, basket_size

Behavioral dimensions:
    distinct_products (product breadth, uses the is_product flag),
    return_rate (negative-quantity lines on normal invoices),
    cancellation_count / cancellation_value / cancellation_rate
    (from C-invoices — the cancellation-behavior dimension)
    has_cancellation (binary flag = any C-invoice; used for clustering)

Temporal:
    tenure (days since first purchase), cadence (days between purchases)

Policy notes:
- Returns (Quantity < 0 on normal invoices) are netted into monetary.
- C-invoices are excluded from RFM but drive the cancellation features.
- Customers with zero real purchases (cancellation-only, 33 in this
  dataset) are excluded: there is no purchasing behavior to segment.
- The snapshot/reference date for recency is max(InvoiceDate).
"""

from __future__ import annotations

import pandas as pd


def build_customer_features(df: pd.DataFrame, snapshot=None) -> pd.DataFrame:
    """Aggregate cleaned transactions into one row per customer.

    Parameters
    ----------
    df : cleaned dataframe from src.data_loader.load_clean()
    snapshot : reference date for recency/tenure (default: max InvoiceDate)

    Returns
    -------
    pd.DataFrame indexed by CustomerID with the features listed in the
    module docstring. Customers with zero real purchases are excluded.
    """
    if snapshot is None:
        snapshot = df["InvoiceDate"].max()

    inv = df["InvoiceNo"].astype(str)
    is_cancel = inv.str.startswith("C")

    sales = df[~is_cancel].copy()
    canc = df[is_cancel].copy()

    sales["line_value"] = sales["Quantity"] * sales["UnitPrice"]
    canc["line_value"] = (canc["Quantity"] * canc["UnitPrice"]).abs()

    # --- RFM + behavioral from real (non-cancellation) invoices ---
    grp = sales.groupby("CustomerID")
    feat = grp.agg(
        frequency=("InvoiceNo", "nunique"),
        monetary=("line_value", "sum"),
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        total_units=("Quantity", "sum"),
        n_lines=("Quantity", "size"),
        return_lines=("Quantity", lambda s: int((s < 0).sum())),
    )

    # Product breadth: only genuine product codes (is_product flag).
    prod = sales[sales["is_product"]]
    dp = prod.groupby("CustomerID")["StockCode"].nunique()
    feat["distinct_products"] = dp

    feat["recency"] = (snapshot - feat["last_purchase"]).dt.days
    feat["tenure"] = (snapshot - feat["first_purchase"]).dt.days
    feat["avg_order_value"] = feat["monetary"] / feat["frequency"]
    feat["basket_size"] = feat["total_units"] / feat["frequency"]
    feat["return_rate"] = feat["return_lines"] / feat["n_lines"]
    feat["cadence"] = feat["tenure"] / feat["frequency"]

    # --- Cancellation-behavior dimension (from C-invoices) ---
    if len(canc):
        cg = canc.groupby("CustomerID").agg(
            cancellation_count=("InvoiceNo", "nunique"),
            cancellation_value=("line_value", "sum"),
        )
        feat = feat.join(cg, how="left")
    feat["cancellation_count"] = feat["cancellation_count"].fillna(0).astype(int)
    feat["cancellation_value"] = feat["cancellation_value"].fillna(0.0)
    feat["cancellation_rate"] = (
        feat["cancellation_count"] / (feat["frequency"] + feat["cancellation_count"])
    )

    # Binary cancellation flag for clustering (denominator-independent —
    # avoids the small-denominator confound of the continuous rate).
    feat["has_cancellation"] = (feat["cancellation_count"] > 0).astype(int)

    feat["distinct_products"] = feat["distinct_products"].fillna(0).astype(int)

    # --- Exclude zero-purchase customers (cancellation-only) ---
    n_zero = int((feat["frequency"] == 0).sum())
    if n_zero:
        feat = feat[feat["frequency"] > 0]
        print(f"[features] excluded {n_zero} cancellation-only customers (no purchases)")

    # Drop helper timestamp columns from the final table.
    feat = feat.drop(columns=["last_purchase", "first_purchase"])

    print(f"[features] built {len(feat):,} customer rows x {feat.shape[1]} features")
    return feat
