"""Verify that normalizing lowercase-suffix StockCodes merges them with valid codes."""
import re

import pandas as pd

df = pd.read_excel("data/Online Retail.xlsx")
sc = df["StockCode"].astype(str)

# Candidate typo codes: digits + single lowercase letter
typo = sc[sc.str.match(r"^\d{4,6}[a-z]$")]
print("Lowercase-suffix typo codes:", typo.nunique(), "distinct,", len(typo), "rows")
print(typo.value_counts().head(10).to_string())

# After uppercasing, how many merge with an existing valid code?
valid = set(sc[sc.str.match(r"^\d{4,6}[A-Z]?$")])
merged = typo.str.upper().isin(valid)
print("\nTypo codes that MERGE with a valid code after uppercasing:",
      int(merged.sum()), "of", len(typo), f"({merged.mean():.1%} of rows)")
print("Typo codes that become NEW (no valid counterpart):",
      int((~typo.str.upper().isin(valid)).sum()), "distinct")

# What about the two-letter and other non-conforming codes?
print("\n=== Non-conforming codes that have digits (likely real products) ===")
has_digits = sc[~sc.str.match(r"^\d{4,6}[A-Z]?$") & sc.str.contains(r"\d")]
print("distinct:", has_digits.nunique(), "rows:", len(has_digits))
print(has_digits.value_counts().head(12).to_string())

# Check 15056BL specifically: does 15056B exist?
print("\nDoes '15056B' exist as a separate code?", "15056B" in set(sc))
print("15056BL description:", df.loc[df.StockCode=="15056BL","Description"].dropna().unique()[:3])