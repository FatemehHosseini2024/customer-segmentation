# Initial EDA — UCI Online Retail (raw transactions)

> Source: `data\Online Retail.xlsx`

> Tool: `eda.py`. **541,909 rows x 8 columns.**

## 1. Structure

| Column | dtype | non-null |
|---|---|---|

| InvoiceNo | object | 541,909 |

| StockCode | object | 541,909 |

| Description | object | 540,455 |

| Quantity | int64 | 541,909 |

| InvoiceDate | datetime64[ns] | 541,909 |

| UnitPrice | float64 | 541,909 |

| CustomerID | float64 | 406,829 |

| Country | object | 541,909 |

## 2. Missing values

Full per-column table also written to `reports\eda_missing.csv`.

| column | count | pct (%) |
| --- | --- | --- |
| InvoiceNo | 0 | 0.000 |
| StockCode | 0 | 0.000 |
| Description | 1454 | 0.270 |
| Quantity | 0 | 0.000 |
| InvoiceDate | 0 | 0.000 |
| UnitPrice | 0 | 0.000 |
| CustomerID | 135080 | 24.930 |
| Country | 0 | 0.000 |

Key observations:

- `CustomerID` is missing in **135,080** rows (~24.9%). These are non-registered / cash-checkout orders that cannot be tied to a customer.

  **Recovery attempt (invoice propagation):** An invoice is a single transaction, so every line of an invoice belongs to the same customer. A row lacking `CustomerID` can therefore be recovered when it shares an `InvoiceNo` with a row that carries one. This is attempted in `src/data_loader.py` (`propagate_customer_id`) and is data completion, not imputation.
  - Recovered via propagation: **0** rows.
  - **Why it recovered nothing — structural, not a failure:** the missing-`CustomerID` rows belong to invoices that are **entirely disjoint** from the customer-ID'd invoices, and no invoice mixes null and non-null `CustomerID` rows. Guest invoices are self-contained, so there is no shared invoice from which to propagate. There is therefore no legitimate way to attribute these rows to a customer; ML imputation would fabricate customer identities and corrupt the segments, so it is not used.
  - **Coverage impact:** dropping these rows removes 135,080 of 541,909 rows (24.9%) but only **34 of 4,372 unique customers (0.8%)** across the full cleaning policy. Since the modelling unit is the customer and every feature is a customer-level aggregation, the excluded rows contribute no usable signal; the retained 4,338 customers still span the bulk of the transaction rows.
- `Description` is missing in **1,454** rows (packing/discount lines). These are **backfilled from the `StockCode` → `Description` map** (`src/data_loader.py` → `backfill_description`): a StockCode is a stable product identifier, so its name is the same everywhere it appears. Recovers **1,342** of 1,454 rows; the **112** remainder are one-off StockCodes that appear exactly once in the workbook with no reference copy. This is product-name completion, not imputation, and does not affect the customer-level feature set.
- `InvoiceDate` has no missing values.

All other columns are fully populated (InvoiceNo, StockCode, Quantity, UnitPrice, Country).
- `Description` (free-text product name, **4,223** unique values, 112 missing after backfill) is retained for validation and human-readable reporting but **excluded from the customer-level feature set**: it is redundant with `StockCode` (~99.7% overlap — the two describe the same product in different forms), it is high-cardinality and noisy (inconsistent spelling of the same product, e.g. `JUMBO BAG RED` vs `JUMBO BAG REDS`), and it is not a customer-behaviour signal at this segmentation granularity. Product breadth is captured via `StockCode` counts instead.

## 3. Duplicates

- Exact duplicate rows: **5,268** — dropped in `src/data_loader.py` (`drop_exact_duplicates`, keep-first). Verified safe: no customer is present only in duplicated rows, so the customer count is unchanged (4,372 before and after).
- Rows sharing (InvoiceNo, StockCode, CustomerID, InvoiceDate): **20,365**. Many are legitimate multi-item invoice lines, but the exact full-row duplicates above are the ones dropped.

## 4. Invoice cancellations

- Invoices prefixed with `C` (order cancellations): **9,288** rows (~1.7%).
- Per the cleaning policy these cancellation invoices are dropped (they represent voided orders, not purchasing behaviour).

## 5. Quantity

| count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 541909.000 | 9.600 | 218.100 | -80995.000 | 1.000 | 3.000 | 10.000 | 80995.000 |

- Negative quantities (returns): **10,624** rows → retained and netted into Monetary value.
- Zero quantities: **0** rows → dropped.
- Largest order: 80,995 units (a bulk / wholesale order; treated as valid).

## 6. UnitPrice

| count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 541909.000 | 4.610 | 96.760 | -11062.060 | 1.250 | 2.080 | 4.130 | 38970.000 |

- Non-positive unit prices (<=0): **2,517** rows → dropped.

## 7. Non-product line items

StockCodes that are not standard numeric product codes (postage, bank charges, DOT adjustments, manuals, etc.):

| StockCode | count |
| --- | --- |
| POST | 1256 |
| DOT | 710 |
| M | 571 |
| 15056BL | 326 |
| C2 | 144 |
| 47566b | 102 |
| 15056n | 84 |
| 72349b | 82 |
| D | 77 |
| 84997d | 75 |
| 84997b | 70 |
| 84509a | 68 |

- Total non-product lines: **5,265** (~1.0%). Detection: a product `StockCode` is defined as 4–6 digits optionally followed by one uppercase letter (`^\d{4,6}[A-Z]?$`); anything else is flagged. This splits into two distinct groups:
  - **Truly non-product codes (no digits): 2,796** — `POST`, `DOT`, `M`, `D`, `S`, `BANK CHARGES`, `AMAZONFEE`, `CRUK`, etc. Postage, bank-charge and manual-adjustment lines that would inflate `distinct_products` and distort Monetary if kept.
  - **Malformed product codes (digits + non-standard suffix): 2,469** — `15056BL`, `C2`, `47566b`, `15056n`, `72349b`, `84997d`, `84997b`, `84509a`, etc. These are almost certainly **real products with casing/typo issues** (e.g. `47566b` is the lowercase form of a valid `47566B`), not junk. A data-quality artifact better normalized than dropped.
- These are noted rather than filtered here: the feature layer defines products via `StockCode` generally, so the cleaning decision is deferred to the feature stage.

## 8. Cardinality / coverage

| dimension | count |
| --- | --- |
| Unique InvoiceNo | 25900 |
| Unique StockCode | 4070 |
| Unique Description | 4223 |
| Unique Country | 38 |
| Unique CustomerID (non-null) | 4372 |

## 9. InvoiceDate range & monthly activity

- Min: `2010-12-01 08:26:00` to Max: `2011-12-09 12:50:00` (~373 days).

Transactions (rows) per month:

| month_end | rows |
| --- | --- |
| 2010-12 | 42481 |
| 2011-01 | 35147 |
| 2011-02 | 27707 |
| 2011-03 | 36748 |
| 2011-04 | 29916 |
| 2011-05 | 37030 |
| 2011-06 | 36874 |
| 2011-07 | 39518 |
| 2011-08 | 35284 |
| 2011-09 | 50226 |
| 2011-10 | 60742 |
| 2011-11 | 84711 |
| 2011-12 | 25525 |
- The data ends mid-December 2011 (truncated final month). The snapshot/reference date used for Recency is `max(InvoiceDate)`.

## 10. Geography (Country)

| country | rows | share_pct |
| --- | --- | --- |
| United Kingdom | 495478 | 91.430 |
| Germany | 9495 | 1.750 |
| France | 8557 | 1.580 |
| EIRE | 8196 | 1.510 |
| Spain | 2533 | 0.470 |
| Netherlands | 2371 | 0.440 |
| Belgium | 2069 | 0.380 |
| Switzerland | 2002 | 0.370 |
| Portugal | 1519 | 0.280 |
| Australia | 1259 | 0.230 |

United Kingdom dominates (~91.4%). With 38 countries but >90% UK, country is a near-constant feature and is excluded from the customer-level feature set (low signal at customer granularity).

## 11. Monetary preview (Quantity x UnitPrice)

| count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 541909.000 | 17.990 | 378.810 | -168469.600 | 3.400 | 9.750 | 17.400 | 168469.600 |

Negative values (returns) and the long positive tail are expected; netting per customer is applied in feature engineering.

## 12. Cleaning policy impact (preview)

Cleaning policy (as implemented in `src/data_loader.py` → `load_clean`): attempt **invoice propagation** to recover `CustomerID` (recovers 0 rows here — guest invoices are disjoint from customer-ID'd invoices); backfill `Description` from the `StockCode` → `Description` map (recovers 1,342 of 1,454); drop **exact duplicate rows** (5,268, keep-first, no customer lost); then drop rows whose `CustomerID` remains unattributable; drop `C`-prefix (cancellation) invoices; drop `UnitPrice <= 0`; drop `Quantity == 0`; retain & net returns (`Quantity < 0`). No columns are dropped at any stage.
| Metric | Raw | After cleaning |
|---|---|---|

| Rows | 541,909 | 392,692 (72.5%) |
| Unique customers | 4,372 | 4,338 |

Per-category drop counts (note: categories overlap, so their sum exceeds the total rows dropped):

| Category dropped | Rows |
|---|---|

| Exact duplicate rows | 5,268 |
| Missing CustomerID | 135,080 |
| C-prefix (cancellation) invoices | 9,288 |
| UnitPrice <= 0 | 2,517 |
| Quantity == 0 | 0 |
