import os
from io import StringIO
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Customer Segmentation App",
    layout="wide",
    page_icon="🧩",
)

st.markdown(
    """
    <style>
    .small-muted { color: #666; font-size: 0.9em; }
    .ok { color: #0a7; }
    .warn { color: #b80; }
    .err { color: #b00; }
    </style>
    """,
    unsafe_allow_html=True,
)


REQUIRED_FEATURES: List[str] = [
    "Age",
    "Income",
    "Total_Spending",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency",
]

MNT_COLUMNS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]


@st.cache_resource(show_spinner=False)
def load_artifacts(models_dir: str = "."):
    """Load scaler and kmeans artifacts with caching and error handling."""
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    kmeans_path = os.path.join(models_dir, "kmeans_model.pkl")

    missing = [p for p in [scaler_path, kmeans_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing required model artifacts: {', '.join(os.path.basename(m) for m in missing)}."
        )

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    return scaler, kmeans


def compute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Age from Year_Birth if Age is missing."""
    if "Age" not in df.columns and "Year_Birth" in df.columns:
        current_year = pd.Timestamp.today().year
        df["Age"] = current_year - pd.to_numeric(df["Year_Birth"], errors="coerce")
    return df


def compute_total_spending(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Total_Spending from Mnt* columns if missing."""
    if "Total_Spending" not in df.columns and any(c in df.columns for c in MNT_COLUMNS):
        present = [c for c in MNT_COLUMNS if c in df.columns]
        df["Total_Spending"] = df[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Attempt to engineer and align columns to REQUIRED_FEATURES.
    Returns (prepared_df, missing_required_columns)
    """
    df = df.copy()

    
    df = compute_age(df)
    df = compute_total_spending(df)

    
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    return df, missing


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def scale_features(df: pd.DataFrame, scaler, feature_order: List[str]) -> np.ndarray:
    
    X = _coerce_numeric(df[feature_order].copy(), feature_order)
    X = X.fillna(X.median())
    return scaler.transform(X)


def predict_clusters(df: pd.DataFrame, scaler, kmeans, feature_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X_scaled = scale_features(df, scaler, feature_order)
    preds = kmeans.predict(X_scaled)
    return preds, X_scaled


def get_cluster_descriptions(n_clusters: int) -> dict:
    
    base = {
        0: "High-value, store‑loyal, very recent",
        1: "Low spend, lapsed browsers",
        2: "Very low spend, price‑sensitive but somewhat recent",
        3: "Previously high‑value omni‑channel, now lapsed",
        4: "Highest income, high spend, store‑heavy, lapsed",
        5: "Mid‑value, digitally engaged, fairly recent"
    }
    return {i: base.get(i, "Segment description pending analysis") for i in range(n_clusters)}


def render_header():
    st.title("Customer Segmentation App 🧩")
    st.caption(
        "Predict customer segment from individual inputs or assign segments in bulk via CSV."
    )


def render_model_status():
    with st.expander("Model artifacts status", expanded=False):
        try:
            scaler, kmeans = load_artifacts()
            st.markdown(
                "✅ <span class='ok'>Artifacts loaded:</span> scaler.pkl, kmeans_model.pkl",
                unsafe_allow_html=True,
            )
            st.write("Expected feature order:", REQUIRED_FEATURES)
            st.write("Number of clusters:", getattr(kmeans, "n_clusters", None))
        except Exception as e:
            st.markdown(
                f"❌ <span class='err'>Artifacts not available:</span> {e}",
                unsafe_allow_html=True,
            )
            st.info(
                "Place trained artifacts (scaler.pkl, kmeans_model.pkl) in the app folder to enable predictions."
            )


# ------------------------------
# UI: Tabs
# ------------------------------
render_header()
render_model_status()

tab_single, tab_batch, tab_profiles, tab_help = st.tabs([
    "Single Prediction",
    "Batch Prediction (CSV)",
    "Cluster Profiles",
    "Help / About",
])


# ------------------------------
# Single Prediction
# ------------------------------
with tab_single:
    st.subheader("Single customer input")
    with st.form("single_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Income", min_value=0, max_value=500000, value=50000, step=1000)
        with col2:
            total_spending = st.number_input(
                "Total Spending (sum of purchases)", min_value=0, max_value=100000, value=1000
            )
            num_web_purchases = st.number_input(
                "Number of Web Purchases", min_value=0, max_value=500, value=10
            )
        with col3:
            num_store_purchases = st.number_input(
                "Number of Store Purchases", min_value=0, max_value=500, value=10
            )
            num_web_visits = st.number_input(
                "Number of Web Visits per month", min_value=0, max_value=200, value=3
            )
        with col4:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=10000, value=30)

        submitted = st.form_submit_button("Predict Segment", width='stretch')

    if submitted:
        try:
            scaler, kmeans = load_artifacts()
            inp = pd.DataFrame(
                {
                    "Age": [age],
                    "Income": [income],
                    "Total_Spending": [total_spending],
                    "NumWebPurchases": [num_web_purchases],
                    "NumStorePurchases": [num_store_purchases],
                    "NumWebVisitsMonth": [num_web_visits],
                    "Recency": [recency],
                }
            )

            preds, X_scaled = predict_clusters(inp, scaler, kmeans, REQUIRED_FEATURES)
            cluster = int(preds[0])

            st.success(f"Predicted Segment: Cluster {cluster}")

            # Distances to all cluster centers (lower = closer)
            try:
                distances = kmeans.transform(X_scaled)[0]
                dist_df = pd.DataFrame({"Cluster": list(range(len(distances))), "Distance": distances})
                st.bar_chart(dist_df.set_index("Cluster"))
            except Exception:
                pass

            # Compare input vs. cluster centroid (approximate inverse-scaled)
            try:
                centers_scaled = kmeans.cluster_centers_
                centers_orig = pd.DataFrame(
                    scaler.inverse_transform(centers_scaled), columns=REQUIRED_FEATURES
                )
                st.caption("Cluster center (approx.) in original feature scale:")
                st.dataframe(
                    centers_orig.round(2), width='stretch'
                )

                st.caption("Your input vs. predicted cluster center:")
                comp = pd.DataFrame(
                    {
                        "Your Input": inp.iloc[0].values,
                        f"Cluster {cluster} Center": centers_orig.iloc[cluster].values,
                    },
                    index=REQUIRED_FEATURES,
                )
                st.dataframe(comp.round(2), width='stretch')
            except Exception:
                pass

            
            desc = get_cluster_descriptions(getattr(kmeans, "n_clusters", 5))
            st.info(f"Segment insight: {desc.get(cluster, 'N/A')}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")



with tab_batch:
    st.subheader("Batch assign segments from CSV")
    st.markdown(
        """
        Upload a CSV with either:
        - The exact required columns: Age, Income, Total_Spending, NumWebPurchases, NumStorePurchases, NumWebVisitsMonth, Recency
        - Or the raw dataset columns (e.g., Year_Birth, Mnt* fields, etc.). The app will derive missing features where possible.
        """
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            st.caption("Preview of uploaded data:")
            st.dataframe(df.head(), width='stretch')

            # Prepare features
            df_prep, missing = prepare_features(df)
            if missing:
                st.warning(
                    "Missing required columns after preparation: "
                    + ", ".join(missing)
                    + ". Please include them or provide raw columns to derive them."
                )
            else:
                try:
                    scaler, kmeans = load_artifacts()
                    preds, _ = predict_clusters(df_prep, scaler, kmeans, REQUIRED_FEATURES)
                    out = df.copy()
                    out["Cluster"] = preds

                    # Display summary and sample
                    st.success("Segments assigned successfully.")
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.caption("Cluster distribution:")
                        st.bar_chart(out["Cluster"].value_counts().sort_index())
                    with c2:
                        st.caption("Sample of results:")
                        st.dataframe(out.head(20), width='stretch')

                    # Download segmented data
                    csv_buf = StringIO()
                    out.to_csv(csv_buf, index=False)
                    st.download_button(
                        label="Download CSV with Cluster Assignments",
                        data=csv_buf.getvalue(),
                        file_name="segmented_customers.csv",
                        mime="text/csv",
                        width='stretch',
                    )
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

    else:
        st.info("Awaiting CSV upload.")


# ------------------------------
# Cluster Profiles
# ------------------------------
with tab_profiles:
    st.subheader("Cluster profiles (approximate)")
    try:
        scaler, kmeans = load_artifacts()
        centers_scaled = kmeans.cluster_centers_
        centers_orig = pd.DataFrame(
            scaler.inverse_transform(centers_scaled), columns=REQUIRED_FEATURES
        )
        centers_orig.index = pd.Index([f"Cluster {i}" for i in range(len(centers_orig))])
        st.dataframe(centers_orig.round(2), width='stretch')

        # Normalized view per feature (min-max across clusters)
        norm = (centers_orig - centers_orig.min()) / (centers_orig.max() - centers_orig.min() + 1e-9)
        st.caption("Normalized profiles (0-1 across clusters per feature):")
        st.dataframe(norm.round(3), width='stretch')

        # Descriptions
        desc = get_cluster_descriptions(getattr(kmeans, "n_clusters", 5))
        st.caption("Segment descriptions:")
        for i in range(len(centers_orig)):
            st.markdown(f"- Cluster {i}: {desc.get(i, 'N/A')}")
    except Exception as e:
        st.warning(f"Unable to display profiles: {e}")


# ------------------------------
# Help / About
# ------------------------------
with tab_help:
    st.subheader("How to use")
    st.markdown(
        """
        - Single Prediction tab: Enter customer attributes and click "Predict Segment".
        - Batch Prediction tab: Upload a CSV. If your file contains raw dataset columns (e.g., Year_Birth, Mnt*), the app will create missing features for you.
        - Cluster Profiles tab: View approximate cluster centers in the original feature space and normalized comparisons.
        
        Notes:
        - Scaling: Inputs are standardized using the training scaler before clustering.
        - Distances: Lower distance indicates a closer match to a cluster center.
        - Descriptions are generic placeholders; align them with your domain analysis.
        """
    )

    st.markdown(
        """
        <span class='small-muted'>
        Built with Streamlit. Artifacts expected in the app directory: <code>scaler.pkl</code> and <code>kmeans_model.pkl</code>.
        Required feature order: <code>Age, Income, Total_Spending, NumWebPurchases, NumStorePurchases, NumWebVisitsMonth, Recency</code>.
        </span>
        """,
        unsafe_allow_html=True,
    )
