"""Builds a deliberately messy demo dataset (missing values, duplicates,
inconsistent formats) so the Level 1 cleaning task has something realistic to work on."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 600
cats = {"Electronics": (150, 700), "Clothing": (15, 120), "Books": (8, 40), "Home": (20, 250), "Sports": (15, 180)}
cat = rng.choice(list(cats), n)
price = [round(rng.uniform(*cats[c]), 2) for c in cat]
dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
fmt = rng.choice(["%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%d %B %Y"], n)
date_str = [d.strftime(f) for d, f in zip(dates, fmt)]
names = rng.choice(["alice smith","BOB JONES","Carla Diaz","dan lee","Eva Brown","frank wu","Grace Kim","henry Ford"], n)
df = pd.DataFrame({
    "order_id": np.arange(1000, 1000 + n),
    "customer_name": [("  " if rng.random()<.1 else "") + x for x in names],
    "age": rng.integers(18, 70, n).astype(float),
    "gender": rng.choice(["M","Male","male","F","Female","female"," f"], n),
    "city": rng.choice(["new york","New York","LONDON","London","paris","Paris","Berlin"], n),
    "order_date": date_str,
    "product_category": [c if rng.random()>.15 else c.lower() for c in cat],
    "quantity": rng.integers(1, 6, n).astype(float),
    "unit_price": price,
}).astype(object)
for col, p in [("age",.07),("gender",.05),("city",.04),("quantity",.06),("unit_price",.05),("order_date",.02)]:
    df.loc[rng.random(n) < p, col] = np.nan
df.loc[rng.choice(n, 4, replace=False), "age"] = [-5, 150, 999, 3]
df.loc[rng.choice(n, 3, replace=False), "quantity"] = "two"
df = pd.concat([df, df.sample(30, random_state=1)], ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)
df.to_csv("raw_data.csv", index=False)
print(df.shape)
