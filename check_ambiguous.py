"""Check descriptions of digit-containing but non-conforming StockCodes."""
import re

import pandas as pd

df = pd.read_excel("data/Online Retail.xlsx")
sc = df["StockCode"].astype(str)

# Digit-containing codes that fail the strict product regex
has_digits = sc[~sc.str.match(r"^\d{4,6}[A-Z]?$") & sc.str.contains(r"\d")]
sub = df[df["StockCode"].isin(has_digits)]
grp = sub.groupby("StockCode")["Description"].agg(
    lambda s: s.dropna().iloc[0] if s.dropna().any() else "(no desc)"
)
counts = sub["StockCode"].value_counts()
out = pd.DataFrame({"rows": counts, "sample_description": grp}).sort_values("rows", ascending=False)
print(out.head(20).to_string())