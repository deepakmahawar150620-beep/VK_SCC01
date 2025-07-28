import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import io
import os

st.set_page_config(page_title="SCC Risk Explorer", layout="wide")
st.title("📊 Stress Corrosion Cracking (SCC) Risk Dashboard")

DATA_CACHE_PATH = "cached_processed_data.parquet"
TOP50_CACHE_PATH = "cached_top50.parquet"

# ---------- Risk Functions ----------
def flag_criteria(df):
    return pd.DataFrame({
        'Stress>60': (df['Hoop stress% of SMYS'] > 60).astype(int),
        'Age>10yrs': (df['Pipe Age'] > 10).astype(int),
        'Temp>38C': (df['Temperature'] > 38).astype(int),
        'Dist≤32km': (df['Distance from Pump(KM)'] <= 32).astype(int),
        'CoatingHighRisk': (~df['CoatingType'].str.upper().isin(['FBE', 'LIQUID EPOXY'])).astype(int),
        'Soil<5000': (df['Soil Resistivity (Ω-cm)'] < 5000).astype(int),
        'OFFPSP>−1.2V': (df['OFF PSP (VE V)'] > -1.2).astype(int)
    })

def compute_risk_score(df):
    flags = flag_criteria(df)
    hs = df['Hoop stress% of SMYS'] / 100.0
    psp = 1 - ((df['OFF PSP (VE V)'] + 2) / 2)
    max_dist = df['Distance from Pump(KM)'].replace(0, np.nan).max() or 1
    dist_norm = (max_dist - df['Distance from Pump(KM)']) / max_dist
    soil_norm = 1 - np.clip(df['Soil Resistivity (Ω-cm)'] / 10000, 0, 1)
    w = {'hs': 0.6, 'psp': 0.3, 'dist': 0.2, 'soil': 0.1}
    score = hs * w['hs'] + psp * w['psp'] + dist_norm * w['dist'] + soil_norm * w['soil']
    return score, flags

# ---------- Clean and Process ----------
def clean_data(df):
    df.columns = df.columns.str.strip()
    df['OFF PSP (VE V)'] = pd.to_numeric(df['OFF PSP (VE V)'], errors='coerce').abs().fillna(0)
    hs = pd.to_numeric(df['Hoop stress% of SMYS'].astype(str).str.replace('%', ''), errors='coerce').fillna(0)
    if hs.max() < 10: hs *= 100
    df['Hoop stress% of SMYS'] = hs
    df['Distance from Pump(KM)'] = pd.to_numeric(df.get('Distance from Pump(KM)', 0), errors='coerce').fillna(1e6)
    df['Pipe Age'] = pd.to_numeric(df.get('Pipe Age', 0), errors='coerce').fillna(0)
    df['Temperature'] = pd.to_numeric(df.get('Temperature', 0), errors='coerce').fillna(0)
    df['Soil Resistivity (Ω-cm)'] = pd.to_numeric(df.get('Soil Resistivity (Ω-cm)', 0), errors='coerce').fillna(1e9)
    df['CoatingType'] = df.get('CoatingType', '').astype(str)
    return df

def process_data(df):
    df = clean_data(df)
    risk_score, flags = compute_risk_score(df)
    df = pd.concat([df, flags], axis=1)
    df['RiskScore'] = risk_score
    df['FlagsSum'] = flags.sum(axis=1)
    df['RiskCategory'] = df['FlagsSum'].apply(lambda x: 'High' if x >= 4 else ('Medium' if x >= 2 else 'Low'))
    top50 = df.sort_values(['RiskScore', 'Hoop stress% of SMYS', 'OFF PSP (VE V)'], ascending=[False, False, False]).head(50)
    return df, top50

# ---------- Button to show uploader ----------
if 'show_uploader' not in st.session_state:
    st.session_state['show_uploader'] = False

if st.button("📤 Upload or Change Excel File"):
    st.session_state['show_uploader'] = True

# ---------- Show uploader only if requested ----------
if st.session_state['show_uploader']:
    uploaded_file = st.file_uploader("📂 Upload Excel File (.xlsx)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        processed_df, top50_df = process_data(df)

        # Save to disk for reuse
        processed_df.to_parquet(DATA_CACHE_PATH)
        top50_df.to_parquet(TOP50_CACHE_PATH)

        # Reset uploader flag
        st.session_state['show_uploader'] = False
        st.success("✅ Excel file processed and saved.")
        st.experimental_rerun()
    else:
        st.stop()

# ---------- Load cached data if available ----------
if os.path.exists(DATA_CACHE_PATH) and os.path.exists(TOP50_CACHE_PATH):
    df = pd.read_parquet(DATA_CACHE_PATH)
    top50 = pd.read_parquet(TOP50_CACHE_PATH)
else:
    st.warning("⚠️ No Excel file found. Please click 'Upload or Change Excel File' to upload one.")
    st.stop()

# ---------- Plot ----------
param = st.selectbox("📌 Select parameter to plot vs Stationing:", [
    'Hoop stress% of SMYS', 'OFF PSP (VE V)', 'Soil Resistivity (Ω-cm)',
    'Distance from Pump(KM)', 'Pipe Age'
])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['Stationing (m)'],
    y=df[param],
    mode='markers',
    marker=dict(
        size=6,
        color=df['RiskScore'],
        colorscale='Reds',
        showscale=True,
        colorbar=dict(title='Risk Score')
    ),
    name=param
))
fig.update_layout(
    title=f"📈 Stationing vs {param} (Color = Risk Score)",
    xaxis_title="Stationing (m)",
    yaxis_title=param,
    template='plotly_white'
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Top 50 Table ----------
st.subheader("🔥 Top 50 High-Risk Locations")
st.dataframe(top50[['Stationing (m)', 'RiskScore', 'RiskCategory',
                    'Hoop stress% of SMYS', 'OFF PSP (VE V)',
                    'Distance from Pump(KM)', 'Soil Resistivity (Ω-cm)',
                    'Pipe Age', 'CoatingType']], use_container_width=True)

# ---------- Download Buttons ----------
csv = top50.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Top 50 CSV", csv, "Top_50_SCC_Risks.csv", "text/csv")

html_buffer = io.StringIO()
pio.write_html(fig, file=html_buffer, include_plotlyjs='cdn')
st.download_button("⬇️ Download Graph as HTML", html_buffer.getvalue(), "SCC_Graph.html", "text/html")
