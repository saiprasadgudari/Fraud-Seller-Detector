import os
import json
from pathlib import Path
import requests
import pandas as pd
import streamlit as st

DEFAULT_API = "http://localhost:8080"
API = os.getenv("API_URL", DEFAULT_API)

try:
    
    API = st.secrets["API_URL"]  
except Exception:
    pass  

st.set_page_config(page_title="Fraud Risk Dashboard", layout="wide")
st.title("Fraudulent Seller Risk (Demo)")

# Load top-20 from artifacts (created by train.py)
top_path = Path("artifacts/top20_test.json")
top20 = []
if top_path.exists():
    try:
        top20 = json.loads(top_path.read_text())
    except Exception:
        top20 = []

#  Layout 
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Top-20 (from test split)")
    if top20:
        df_top = pd.DataFrame(top20)
        # normalize common column names
        if "prob" in df_top.columns and "fraud_prob" not in df_top.columns:
            df_top = df_top.rename(columns={"prob": "fraud_prob"})
        st.dataframe(df_top, use_container_width=True, height=420)
    else:
        st.info("Run training to create artifacts/top20_test.json")

with col2:
    st.subheader("Score a seller")
    sid = st.text_input("seller_id", placeholder="e.g., S00000")
    if st.button("Score"):
        if not sid:
            st.warning("Enter a seller_id first.")
        else:
            try:
                r = requests.post(f"{API}/score", json={"seller_id": sid}, timeout=10)
                if r.status_code == 200:
                    st.success(r.json())
                else:
                    st.error(f"{r.status_code}: {r.text}")
            except Exception as e:
                st.exception(e)

st.divider()

#  Explain panel 
st.subheader("Explain a prediction")
ex_col1, ex_col2 = st.columns([2, 1], gap="large")

with ex_col1:
    options = [row.get("seller_id") for row in top20] if top20 else []
    default_sid = options[0] if options else ""
    sid_exp = st.text_input("seller_id to explain", value=default_sid, key="sid_explain")
    k = st.slider("Top K features", 5, 20, 10, step=1)
    btn = st.button("Explain")
    if btn:
        if not sid_exp:
            st.warning("Enter a seller_id first.")
        else:
            try:
                r = requests.post(f"{API}/explain", json={"seller_id": sid_exp, "top_k": k}, timeout=20)
                if r.status_code == 200:
                    payload = r.json()
                    st.write({"seller_id": payload["seller_id"], "fraud_prob": round(payload["fraud_prob"], 6)})

                    feats = pd.DataFrame(payload["features"])
                    if not feats.empty:
                        feats["abs_contribution"] = feats["contribution"].abs()
                        st.caption("Top contributors")
                        st.dataframe(feats[["feature", "value", "contribution"]], use_container_width=True, height=360)
                        st.bar_chart(feats.set_index("feature")["contribution"])
                    else:
                        st.info("No explanation available (model/SHAP not supported).")
                else:
                    st.error(f"{r.status_code}: {r.text}")
            except Exception as e:
                st.exception(e)

with ex_col2:
    st.markdown("**Submit feedback** (label this seller)")
    sid_fb = st.text_input("seller_id", value=default_sid, key="sid_feedback")
    label = st.selectbox("Label", options=[("Not fraud (0)", 0), ("Fraud (1)", 1)], format_func=lambda x: x[0])
    note = st.text_area("Note (optional)", placeholder="Why you think this label is correct…")
    if st.button("Send feedback"):
        if not sid_fb:
            st.warning("Enter a seller_id first.")
        else:
            try:
                payload = {"seller_id": sid_fb, "label": label[1], "source": "ui", "note": note or None}
                r = requests.post(f"{API}/feedback", json=payload, timeout=10)
                if r.status_code == 200:
                    st.success(f"Thanks! Saved → {r.json().get('written')}")
                else:
                    st.error(f"{r.status_code}: {r.text}")
            except Exception as e:
                st.exception(e)

st.divider()
st.caption(f"API: {API} • Change via .streamlit/secrets.toml → API_URL")
