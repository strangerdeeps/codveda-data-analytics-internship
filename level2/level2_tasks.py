# %% [markdown]
# # Codveda Data Analytics Internship - Level 2 (Intermediate)
# **Task 1:** Regression Analysis  **Task 2:** Clustering Analysis (K-Means)
#
# Datasets: scikit-learn's built-in Diabetes (regression) and Wine (clustering) datasets.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")
os.makedirs("plots", exist_ok=True)

# %% [markdown]
# ## Task 1: Regression Analysis
# Predict disease progression (`target`) one year after baseline.
# First a simple linear regression (one predictor, BMI), then a multiple regression for comparison.

# %%
data = load_diabetes(as_frame=True)
df = data.frame
print(df.shape)
print(df.describe().round(3))

# %% [markdown]
# ### Split, fit, evaluate (simple regression: bmi -> target)

# %%
X = df[["bmi"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
mse = mean_squared_error(y_test, pred)
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient (bmi): {model.coef_[0]:.2f}")
print(f"R-squared: {r2:.3f}")
print(f"MSE: {mse:.2f}   RMSE: {np.sqrt(mse):.2f}")

# %%
plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, alpha=0.7, label="Actual (test)")
xs = np.linspace(X["bmi"].min(), X["bmi"].max(), 100).reshape(-1, 1)
plt.plot(xs, model.predict(pd.DataFrame(xs, columns=["bmi"])), color="red", label="Regression line")
plt.xlabel("BMI (standardized)"); plt.ylabel("Disease progression")
plt.title("Simple linear regression: BMI vs disease progression")
plt.legend(); plt.tight_layout(); plt.savefig("plots/regression_simple.png", dpi=150); plt.close()

# %% [markdown]
# ### Multiple regression (all features) for comparison

# %%
Xa = df.drop(columns="target")
Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa, y, test_size=0.2, random_state=42)
multi = LinearRegression().fit(Xa_train, ya_train)
pa = multi.predict(Xa_test)
print(f"Multiple regression  R2: {r2_score(ya_test, pa):.3f}   MSE: {mean_squared_error(ya_test, pa):.2f}")
print(pd.Series(multi.coef_, index=Xa.columns).sort_values().round(1))

plt.figure(figsize=(6, 6))
plt.scatter(ya_test, pa, alpha=0.7)
lims = [y.min(), y.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Multiple regression: actual vs predicted")
plt.tight_layout(); plt.savefig("plots/regression_multiple.png", dpi=150); plt.close()

# %% [markdown]
# **Interpretation:** sklearn's diabetes features are pre-scaled, so the BMI coefficient (~999) is per
# unit of that scaled feature, not per BMI point. Its sign is positive: higher BMI is associated with
# higher disease progression. BMI alone gives R-squared = 0.23 (RMSE ~ 64); using all ten features
# raises R-squared to 0.45 (MSE 2900 vs 4062), so BMI matters but the outcome depends on several
# factors (s5, bmi, s2 and bp carry the largest positive coefficients).

# %% [markdown]
# ## Task 2: Clustering Analysis (K-Means)

# %%
wine = load_wine(as_frame=True)
W = wine.frame.drop(columns="target")
print(W.shape)

# 1. standardize
Xs = StandardScaler().fit_transform(W)

# 2. elbow method + silhouette
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Xs, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow method")
ax[0].set_xlabel("Number of clusters (k)"); ax[0].set_ylabel("Inertia (WCSS)")
ax[1].plot(list(ks), sil, "o-", color="green"); ax[1].set_title("Silhouette score")
ax[1].set_xlabel("Number of clusters (k)"); ax[1].set_ylabel("Score")
plt.tight_layout(); plt.savefig("plots/kmeans_elbow.png", dpi=150); plt.close()
print(pd.DataFrame({"k": list(ks), "inertia": np.round(inertia, 1), "silhouette": np.round(sil, 3)}))

# %%
K = 3  # silhouette peaks at k=3 and the elbow flattens after it (also matches the 3 known wine cultivars)
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(Xs)
W["cluster"] = km.labels_

# 3. 2D visualization via PCA
pcs = PCA(n_components=2).fit_transform(Xs)
plt.figure(figsize=(7, 5))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=W["cluster"], palette="Set1", s=60)
centers = PCA(n_components=2).fit(Xs).transform(km.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=250, c="black", label="Centroids")
plt.title(f"K-Means clusters (k={K}) in PCA space"); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(); plt.tight_layout(); plt.savefig("plots/kmeans_clusters.png", dpi=150); plt.close()

print("Cluster sizes:\n", W["cluster"].value_counts().sort_index())
print("\nCluster profiles (feature means):\n", W.groupby("cluster").mean().round(2).T)
print("\nSilhouette at k=3:", round(silhouette_score(Xs, km.labels_), 3))
