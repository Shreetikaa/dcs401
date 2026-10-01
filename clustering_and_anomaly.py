"""
clustering_and_anomaly.py
---------------------------
Two things, on the same customer data:

1. Clustering (K-Means) -- groups customers into behavior segments based
   on how recently, how often, and how much they buy (RFM).
2. Anomaly detection (DBSCAN) -- finds customers whose behavior is very
   different from everyone else (unusually high spenders, one-off buyers
   with huge orders, etc.) without being told in advance what "unusual"
   looks like.

Needs: data/online_retail_cleaned.csv (the output of load_and_clean.py)

Run with:  python clustering_and_anomaly.py
Output:    customer_clusters.png, customer_anomalies.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

CLEANED_CSV_PATH = "data/online_retail_cleaned.csv"

# ---------------------------------------------------------------------------
# Step 1: Load the cleaned data
# ---------------------------------------------------------------------------
df = pd.read_csv(CLEANED_CSV_PATH)
print(f"Loaded {len(df):,} cleaned rows")

# Every single row is used -- no exceptions. Returns are included (a return's
# line_total is already negative, so it nets each customer's real spend).
# Rows with no customer_id can't be tied to one specific person, but they're
# still real transactions, so instead of dropping them they're folded into one
# shared "Unknown (guest checkouts)" entry -- every row in the file ends up
# inside some group in the table below.
UNKNOWN_ID = -1
UNKNOWN_LABEL = "Unknown (guest checkouts)"

n_unknown_rows = df["customer_id"].isna().sum()
print(f"{n_unknown_rows:,} rows have no customer_id -- grouped into a single "
      f"'{UNKNOWN_LABEL}' entry rather than dropped, so all {len(df):,} rows are used")
df["customer_id"] = df["customer_id"].fillna(UNKNOWN_ID).astype(int)
df["invoice_date"] = pd.to_datetime(df["invoice_date"])

# ---------------------------------------------------------------------------
# Step 2: Build one row per customer -- Recency, Frequency, Monetary (RFM)
# ---------------------------------------------------------------------------
as_of = df["invoice_date"].max()

rfm = df.groupby("customer_id").agg(
    recency_days=("invoice_date", lambda d: (as_of - d.max()).days),
    frequency=("invoice_no", "nunique"),
    monetary=("line_total", "sum"),  # net spend: purchases minus returns
    first_purchase=("invoice_date", "min"),
    last_purchase=("invoice_date", "max"),
).reset_index()
rfm["customer_label"] = rfm["customer_id"].apply(
    lambda cid: UNKNOWN_LABEL if cid == UNKNOWN_ID else str(cid)
)

# A few customers return more than they keep, so their net spend is 0 or
# negative -- real, and kept in the data and every table. Log scale can't
# plot zero/negative though, so charts use a separate floor-clipped column
# (monetary_plot) just for the y-position; hover/tables always show the
# true net monetary value.
n_non_positive = (rfm["monetary"] <= 0).sum()
print(f"{n_non_positive:,} customers have net spend <= £0 after returns "
      f"(kept in the data; shown at the chart floor since log scale can't plot them)")
rfm["monetary_plot"] = rfm["monetary"].clip(lower=1)

# --- extra features derived from invoice dates, beyond just recency --------
# tenure_days: how long this customer has been buying at all (first to last
# purchase). A customer who bought once and vanished looks very different
# from one who has been active for a year, even with the same recency.
rfm["tenure_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days

# avg_days_between_purchases: how spaced out their orders are. A customer
# with 10 orders crammed into 2 days behaves very differently from one with
# 10 orders spread across a year -- this catches that, which recency/
# frequency/monetary alone can't.
rfm["avg_days_between_purchases"] = rfm.apply(
    lambda r: r["tenure_days"] / (r["frequency"] - 1) if r["frequency"] > 1 else 0,
    axis=1,
)

print(f"\nBuilt RFM + date-based feature table for {len(rfm):,} customers")
print(rfm[["recency_days", "frequency", "monetary", "tenure_days", "avg_days_between_purchases"]].describe().round(1))

# ---------------------------------------------------------------------------
# Step 3: Scale the features
# ---------------------------------------------------------------------------
# All 5 features are on very different scales (days vs. order counts vs.
# money), so each is standardized to mean 0, std 1 before distance-based
# algorithms (K-Means, DBSCAN) can compare them fairly.
FEATURE_COLS = ["recency_days", "frequency", "monetary", "tenure_days", "avg_days_between_purchases"]
features = rfm[FEATURE_COLS]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ---------------------------------------------------------------------------
# Step 4: Clustering -- K-Means groups customers into behavior segments
# ---------------------------------------------------------------------------
N_CLUSTERS = 4
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
rfm["cluster"] = kmeans.fit_predict(features_scaled)

print(f"\nK-Means found {N_CLUSTERS} customer clusters:")
cluster_profile = rfm.groupby("cluster")[FEATURE_COLS].mean().round(1)
cluster_profile["customers"] = rfm["cluster"].value_counts().sort_index()
print(cluster_profile)

# Plot the clusters (Recency vs Monetary, colored by cluster)
plt.figure(figsize=(9, 6))
scatter = plt.scatter(rfm["recency_days"], rfm["monetary_plot"], c=rfm["cluster"], cmap="tab10", alpha=0.6)
plt.yscale("log")
plt.title(f"Customer Clusters (K-Means, k={N_CLUSTERS})")
plt.xlabel("Recency (days since last purchase)")
plt.ylabel("Net spend, purchases minus returns (log scale)")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.tight_layout()
plt.savefig("customer_clusters.png")
print("\nSaved chart: customer_clusters.png")

# ---------------------------------------------------------------------------
# Step 5: Anomaly detection -- DBSCAN flags customers who don't fit any group
# ---------------------------------------------------------------------------
# DBSCAN groups points that are close together and labels anything that
# isn't close to any group as an outlier (label -1). Unlike K-Means, it
# doesn't force every point into a cluster -- that's what makes it useful
# for finding anomalies instead of just segments.
dbscan = DBSCAN(eps=0.5, min_samples=10)
rfm["anomaly_label"] = dbscan.fit_predict(features_scaled)
rfm["is_anomaly"] = rfm["anomaly_label"] == -1

n_anomalies = rfm["is_anomaly"].sum()
print(f"\nDBSCAN flagged {n_anomalies:,} anomalous customers out of {len(rfm):,} "
      f"({100 * n_anomalies / len(rfm):.2f}%)")

print("\nTop 10 anomalies by spend:")
print(rfm[rfm["is_anomaly"]].sort_values("monetary", ascending=False)
      .head(10)[["customer_label", "recency_days", "frequency", "monetary", "tenure_days", "avg_days_between_purchases"]])

# Plot the anomalies (Recency vs Monetary, anomalies highlighted in red)
plt.figure(figsize=(9, 6))
normal = rfm[~rfm["is_anomaly"]]
anomalies = rfm[rfm["is_anomaly"]]
plt.scatter(normal["recency_days"], normal["monetary_plot"], color="steelblue", alpha=0.5, label="Normal")
plt.scatter(anomalies["recency_days"], anomalies["monetary_plot"], color="red", alpha=0.8, label="Anomaly")
plt.yscale("log")
plt.title("Customer Anomalies (DBSCAN)")
plt.xlabel("Recency (days since last purchase)")
plt.ylabel("Net spend, purchases minus returns (log scale)")
plt.legend()
plt.tight_layout()
plt.savefig("customer_anomalies.png")
print("Saved chart: customer_anomalies.png")
