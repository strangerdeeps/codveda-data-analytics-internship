# %% [markdown]
# # Codveda Data Analytics Internship - Level 1 (Basic)
# **Task 1:** Data Cleaning & Preprocessing  **Task 2:** Exploratory Data Analysis
#
# Set `RAW_PATH` to your own raw CSV to rerun everything on a different dataset
# (column names in the cleaning section may need adjusting).

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
RAW_PATH = "raw_data.csv"
os.makedirs("plots", exist_ok=True)

# %% [markdown]
# ## Task 1: Data Cleaning & Preprocessing
# ### 1. Load the dataset

# %%
df = pd.read_csv(RAW_PATH)
print("Shape:", df.shape)
print(df.head(), "\n")
print(df.dtypes)

# %% [markdown]
# ### 2. Identify missing values and duplicates

# %%
print("Missing values per column:\n", df.isna().sum(), "\n")
print("Duplicate rows:", df.duplicated().sum())

# %% [markdown]
# ### 3. Remove duplicates and standardize formats

# %%
n0 = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {n0 - len(df)} duplicate rows")

# text columns: trim whitespace, consistent case
df["customer_name"] = df["customer_name"].str.strip().str.title()
df["city"] = df["city"].str.strip().str.title()
df["product_category"] = df["product_category"].str.strip().str.title()

# categorical: unify gender labels
gender_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
df["gender"] = df["gender"].str.strip().str.lower().map(gender_map)

# dates: mixed formats -> one datetime format
df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", errors="coerce")

# numeric: coerce non-numeric entries to NaN
for col in ["age", "quantity", "unit_price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# impossible ages -> missing
df.loc[(df["age"] < 16) | (df["age"] > 100), "age"] = np.nan

# %% [markdown]
# ### 4. Handle missing values
# - Numeric columns: median imputation (robust to outliers)
# - Categorical columns: mode imputation
# - Rows missing the order date (key field) are dropped

# %%
print("Missing before:\n", df.isna().sum(), "\n")
df = df.dropna(subset=["order_date"])
for col in ["age", "quantity", "unit_price"]:
    df[col] = df[col].fillna(df[col].median())
for col in ["gender", "city", "product_category", "customer_name"]:
    df[col] = df[col].fillna(df[col].mode()[0])
df["age"] = df["age"].round().astype(int)
df["quantity"] = df["quantity"].round().astype(int)
df["total_amount"] = (df["quantity"] * df["unit_price"]).round(2)
print("Missing after:\n", df.isna().sum())

df.to_csv("cleaned_data.csv", index=False)
print("\nCleaned shape:", df.shape)
print(df.head())

# %% [markdown]
# ## Task 2: Exploratory Data Analysis
# ### 1. Summary statistics

# %%
num_cols = ["age", "quantity", "unit_price", "total_amount"]
summary = df[num_cols].agg(["mean", "median", "std", "min", "max"]).T
summary["mode"] = [df[c].mode()[0] for c in num_cols]
print(summary.round(2))
summary.round(2).to_csv("summary_statistics.csv")

# %% [markdown]
# ### 2. Distributions: histograms, boxplots, scatter plot

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, c in zip(axes, num_cols):
    sns.histplot(df[c], kde=True, ax=ax)
    ax.set_title(f"Distribution of {c}")
plt.tight_layout(); plt.savefig("plots/histograms.png", dpi=150); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df, x="product_category", y="total_amount", ax=axes[0])
axes[0].set_title("Order total by category"); axes[0].tick_params(axis="x", rotation=30)
sns.boxplot(data=df, x="gender", y="age", ax=axes[1])
axes[1].set_title("Age by gender")
plt.tight_layout(); plt.savefig("plots/boxplots.png", dpi=150); plt.close()

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="unit_price", y="total_amount", hue="product_category", alpha=0.7)
plt.title("Unit price vs order total")
plt.tight_layout(); plt.savefig("plots/scatter.png", dpi=150); plt.close()

# %% [markdown]
# ### 3. Correlations between numerical features

# %%
corr = df[num_cols].corr()
print(corr.round(2))
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix")
plt.tight_layout(); plt.savefig("plots/correlation_heatmap.png", dpi=150); plt.close()

# %% [markdown]
# ### 4. Trends and patterns

# %%
monthly = df.set_index("order_date")["total_amount"].resample("MS").sum()
print(monthly.round(0))
print("\nRevenue by category:\n", df.groupby("product_category")["total_amount"].sum().round(0).sort_values(ascending=False))
print("\nRevenue by city:\n", df.groupby("city")["total_amount"].sum().round(0).sort_values(ascending=False))
