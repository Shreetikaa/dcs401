"""
app.py
-------
Streamlit dashboard for the clustering + anomaly detection script.

Needs: data/online_retail_cleaned.csv (the output of load_and_clean.py)

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

CLEANED_CSV_PATH = "data/online_retail_cleaned.csv"

# ---------------------------------------------------------------------------
# Page setup + visual style
#
# Design idea: this is a ledger of real sales transactions, so the dashboard
# borrows from that -- a serif heading (like a ledger title), monospaced
# figures (like columns of numbers in a register), and a flat, ruled layout
# instead of rounded, shadowed "app" cards. Accent colors come from a muted
# ink-and-brass palette rather than default dashboard blue.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Customer Behavior Ledger", page_icon="\U0001F4D2", layout="wide")

INK = "#20283A"
MUTED = "#6B7482"
RULE = "#D8DCE1"
FOREST = "#2F5D50"
BRASS = "#A9792C"
BRICK = "#A8463A"
PANEL = "#F6F7F5"

SEGMENT_PALETTE = ["#2F5D50", "#4F7D63", "#7FA07A", "#A9792C", "#C7A45C", "#8C7A63", "#5B6B73", "#3D4A52"]
TYPICAL_COLOR = "#B8BEC7"

# System font stacks only -- no external font fetch, so headings and figures
# render consistently even without internet access to a font CDN.
SERIF = "Georgia, 'Iowan Old Style', 'Palatino Linotype', serif"
SANS = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
MONO = "'Consolas', 'SFMono-Regular', Menlo, Monaco, monospace"

st.markdown(
    f"""
    <style>
    #MainMenu, footer, div[data-testid="stToolbar"] {{ visibility: hidden; }}

    /* Force a light theme regardless of the browser/OS dark-mode setting --
       otherwise ink-dark text can land on a dark background and disappear. */
    :root, html, body {{
        color-scheme: light;
    }}
    html, body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"],
    section.main {{
        background-color: #FFFFFF !important;
    }}
    .stApp, .stApp * {{
        color: {INK};
    }}
    html, body, [class*="css"] {{
        font-family: {SANS};
        color: {INK};
    }}
    .block-container {{
        padding-top: 2.2rem;
        max-width: 1180px;
    }}

    .ledger-head {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        flex-wrap: wrap;
        row-gap: 0.3rem;
    }}
    .ledger-title {{
        font-family: {SERIF};
        font-size: 2.1rem;
        font-weight: 700;
        color: {INK};
    }}
    .ledger-meta {{
        font-family: {MONO};
        font-size: 0.85rem;
        color: {MUTED};
    }}
    .ledger-subtitle {{
        font-size: 1rem;
        color: {MUTED};
        margin-top: 0.15rem;
    }}
    .ledger-rule {{
        border: none;
        border-top: 1px dashed {RULE};
        margin: 1.1rem 0 1.25rem 0;
    }}

    div[data-testid="stMetricValue"] {{
        font-family: {MONO};
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 3px !important;
        border-color: {RULE} !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 1.6rem;
        border-bottom: 1px solid {RULE};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.98rem;
        font-weight: 600;
        padding: 0.5rem 0.1rem;
        color: {MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        color: {INK} !important;
    }}

    .section-head {{
        font-family: {SERIF};
        font-size: 1.35rem;
        font-weight: 600;
        color: {INK};
        margin-bottom: 0.1rem;
    }}
    .section-note {{
        font-size: 0.86rem;
        color: {MUTED};
        margin-bottom: 0.9rem;
    }}

    div[data-testid="stSlider"] div[style*="translate(-50%"] {{
        background-color: {FOREST} !important;
        border-color: {FOREST} !important;
    }}
    div[data-testid="stSliderThumbValue"] {{
        color: {FOREST} !important;
    }}
    div[data-testid="stSlider"] [role="group"] > div > div:first-child:not([style]) {{
        background: {RULE} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="ledger-head">'
    '<div class="ledger-title">Customer Behavior Ledger</div>'
    '</div>'
    '<div class="ledger-subtitle">Every customer\'s order history, grouped into behavior '
    'segments and checked for anything unusual.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load + prepare data (needed before the expander below, which reports how
# many rows were excluded)
# ---------------------------------------------------------------------------
UNKNOWN_ID = -1
UNKNOWN_LABEL = "Unknown (guest checkouts)"


@st.cache_data
def load_rfm():
    df = pd.read_csv(CLEANED_CSV_PATH)

    # Every single row is used -- no exceptions. Returns are included (a
    # return's line_total is already negative, so it nets each customer's
    # real spend). Rows with no customer_id can't be tied to one specific
    # person, but they're real transactions, so instead of dropping them
    # they're folded into one shared "Unknown (guest checkouts)" entry --
    # every row in the file ends up inside some group in this table.
    n_unknown_rows = df["customer_id"].isna().sum()
    df["customer_id"] = df["customer_id"].fillna(UNKNOWN_ID).astype(int)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    as_of = df["invoice_date"].max()
    rfm = df.groupby("customer_id").agg(
        recency_days=("invoice_date", lambda d: (as_of - d.max()).days),
        frequency=("invoice_no", "nunique"),
        monetary=("line_total", "sum"),  # net spend: purchases minus returns
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
    ).reset_index()

    # extra features derived from invoice dates, beyond just recency
    rfm["tenure_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["avg_days_between_purchases"] = rfm.apply(
        lambda r: r["tenure_days"] / (r["frequency"] - 1) if r["frequency"] > 1 else 0,
        axis=1,
    )

    rfm["customer_label"] = rfm["customer_id"].apply(
        lambda cid: UNKNOWN_LABEL if cid == UNKNOWN_ID else str(cid)
    )

    # A few customers return more than they keep, so net spend is £0 or
    # negative -- real, and kept everywhere. Log-scale charts can't plot
    # zero/negative, so this floor-clipped column is only for the y-position;
    # hover text and tables always show the true monetary value.
    rfm["monetary_plot"] = rfm["monetary"].clip(lower=1)

    return rfm, df["invoice_date"].min(), df["invoice_date"].max(), n_unknown_rows, len(df)


FEATURE_COLS = ["recency_days", "frequency", "monetary", "tenure_days", "avg_days_between_purchases"]
FRIENDLY_NAMES = {
    "recency_days": "Days since last order",
    "frequency": "Total orders",
    "monetary": "Net spend (£)",
    "tenure_days": "Days as a customer",
    "avg_days_between_purchases": "Avg. days between orders",
}

rfm, date_min, date_max, n_unknown_rows, n_total_rows = load_rfm()

# Scale features once -- shared by both clustering and anomaly detection
features = rfm[FEATURE_COLS]
features_scaled = StandardScaler().fit_transform(features)

with st.expander("Reading this ledger — what the 5 numbers mean"):
    st.markdown(
        "Every customer is described by 5 numbers, built from their order history:\n\n"
        "- **Recency** — days since their last order\n"
        "- **Frequency** — how many orders they've placed\n"
        "- **Monetary** — net spend: purchases minus returns\n"
        "- **Tenure** — how long they've been a customer\n"
        "- **Order spacing** — how evenly spread out their orders are\n\n"
        "**Customer segments** groups customers who look similar on these "
        "5 numbers. **Unusual customers** flags anyone who doesn't "
        "resemble any group — potential VIPs, one-off bulk buyers, or data to "
        "double-check.\n\n"
        f"All {n_total_rows:,} rows of cleaned order history are used here — "
        "none are left out. Returns are included too, netted against each "
        "customer's spend instead of being dropped. "
        f"{n_unknown_rows:,} of those rows have no customer ID (guest "
        "checkouts), so they can't be tied to one specific person — rather "
        "than excluding them, they're grouped into a single **"
        f"{UNKNOWN_LABEL}**"
        " entry, which shows up in the tables and charts like any other "
        "customer. A few real customers return more than they keep, landing "
        "at £0 net spend or below; they're kept in the data and tables, just "
        "shown at the chart's floor since a log scale can't plot zero or "
        "negative values."
    )

st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Customer segments", "Unusual customers"])

# --------------------------------------------------------------- Clustering
with tab1:
    st.markdown('<div class="section-head">Group customers who behave alike</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Method: K-Means clustering on the 5 features above</div>',
                unsafe_allow_html=True)

    with st.container(border=True):
        n_clusters = st.slider(
            "How many segments should we look for?", min_value=2, max_value=8, value=4,
            help="K-Means needs to be told how many groups to split customers into. "
                 "Try a few values and see which gives the most distinct, sensible groups.",
        )

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster_id"] = kmeans.fit_predict(features_scaled)

    # Rank clusters by average spend so labels are meaningful regardless of
    # how many clusters were requested, or which raw cluster id sklearn gave.
    TIER_LABELS = [
        "Top spenders", "High-value customers", "Steady regulars", "Occasional buyers",
        "Light buyers", "At-risk customers", "Lapsed customers", "Rare buyers",
    ]
    rank_by_spend = rfm.groupby("cluster_id")["monetary"].mean().sort_values(ascending=False)
    tier_map = {cid: TIER_LABELS[i] for i, cid in enumerate(rank_by_spend.index)}
    rfm["segment"] = rfm["cluster_id"].map(tier_map)
    color_map = {TIER_LABELS[i]: SEGMENT_PALETTE[i % len(SEGMENT_PALETTE)] for i in range(n_clusters)}

    fig1 = px.scatter(
        rfm, x="recency_days", y="monetary_plot", color="segment", size="frequency",
        category_orders={"segment": [t for t in TIER_LABELS if t in color_map]},
        color_discrete_map=color_map,
        log_y=True,
        labels={"recency_days": "Days since last order", "monetary_plot": "Net spend (£)", "segment": "Segment", "customer_label": "Customer"},
        hover_data={"customer_label": True, "customer_id": False, "frequency": True, "monetary": ":,.0f", "monetary_plot": False},
    )
    fig1.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=SANS, color=INK),
        legend_title_text="Segment", margin=dict(t=20),
    )
    fig1.update_xaxes(gridcolor=RULE, zerolinecolor=RULE)
    fig1.update_yaxes(gridcolor=RULE, zerolinecolor=RULE)
    st.plotly_chart(fig1, use_container_width=True, theme=None)
    st.caption("Each dot is one customer. Bubble size = number of orders placed. Hover a dot for details.")

    st.markdown('<div class="section-head" style="font-size:1.1rem;">What each segment looks like</div>',
                unsafe_allow_html=True)
    profile = rfm.groupby("segment")[FEATURE_COLS].mean().reindex(
        [t for t in TIER_LABELS if t in color_map]
    ).round(1)
    profile["Customers"] = rfm["segment"].value_counts()
    profile = profile.rename(columns=FRIENDLY_NAMES)
    st.dataframe(
        profile,
        use_container_width=True,
        column_config={
            "Net spend (£)": st.column_config.NumberColumn(format="£%,.0f"),
        },
    )

# ---------------------------------------------------------- Anomaly Detection
with tab2:
    st.markdown('<div class="section-head">Find customers who don\'t resemble anyone else</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-note">Method: DBSCAN density-based outlier detection</div>',
                unsafe_allow_html=True)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider(
                "Sensitivity", 0.1, 2.0, 0.5, 0.1,
                help="Lower = stricter, flags more customers as unusual. "
                     "Higher = more lenient, only flags the most extreme cases.",
            )
        with col2:
            min_samples = st.slider(
                "Minimum typical group size", 3, 30, 10,
                help="How many similar customers it takes to count as a 'normal' group. "
                     "Anyone not part of a group this size or larger is flagged as unusual.",
            )

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    rfm["anomaly_label"] = dbscan.fit_predict(features_scaled)
    rfm["status"] = rfm["anomaly_label"].apply(lambda x: "Unusual" if x == -1 else "Typical")

    n_anomalies = (rfm["status"] == "Unusual").sum()
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Unusual customers found", f"{n_anomalies:,}",
                   f"{100 * n_anomalies / len(rfm):.1f}% of all customers")
    with m2:
        st.metric("Typical customers", f"{len(rfm) - n_anomalies:,}")

    fig2 = px.scatter(
        rfm, x="recency_days", y="monetary_plot", color="status",
        color_discrete_map={"Typical": TYPICAL_COLOR, "Unusual": BRICK},
        category_orders={"status": ["Typical", "Unusual"]},
        log_y=True,
        labels={"recency_days": "Days since last order", "monetary_plot": "Net spend (£)", "status": "Status", "customer_label": "Customer"},
        hover_data={"customer_label": True, "customer_id": False, "frequency": True, "monetary": ":,.0f", "monetary_plot": False},
    )
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=SANS, color=INK),
        legend_title_text="", margin=dict(t=20),
    )
    fig2.update_xaxes(gridcolor=RULE, zerolinecolor=RULE)
    fig2.update_yaxes(gridcolor=RULE, zerolinecolor=RULE)
    st.plotly_chart(fig2, use_container_width=True, theme=None)
    st.caption("Dark-red dots are customers DBSCAN couldn't fit into any typical group — "
               "often big one-off spenders, or buying patterns worth a second look.")

    st.markdown('<div class="section-head" style="font-size:1.1rem;">Highest-spending unusual customers</div>',
                unsafe_allow_html=True)
    unusual_table = (
        rfm[rfm["status"] == "Unusual"]
        .sort_values("monetary", ascending=False)
        .head(15)[["customer_label"] + FEATURE_COLS]
        .rename(columns=FRIENDLY_NAMES)
        .rename(columns={"customer_label": "Customer"})
    )
    st.dataframe(
        unusual_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Net spend (£)": st.column_config.NumberColumn(format="£%,.0f"),
        },
    )
