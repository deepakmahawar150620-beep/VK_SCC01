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

# 1️⃣ Always show upload option
uploaded_file = st.file_uploader("📂 Upload Excel File (.xlsx)", type=["xlsx"])

# 2️⃣ If uploaded, process and cache
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Clean & process
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

    # Risk flags
    flags = pd.DataFrame({
        'Stress>60': (df['Hoop stress% of SMYS'] > 60).astype(int),
        'Age>10yrs': (df['Pipe Age'] > 10).astype(int),
        'Temp>38C': (df['Temperature'] > 38).astype(int),
        'Dist≤32km': (df['Distance from Pump(KM)'] <= 32).astype(int),
        'CoatingHighRisk': (~df['CoatingType'].str.upper().isin(['FBE', 'LIQUID EPOXY'])).astype(int),
        'Soil<5000': (df['Soil Resistivity (Ω-cm)'] < 5000).astype(int),
        'OFFPSP>−1.2V': (df['OFF PSP (VE V)'] > -1.2).astype(int)
    })

    # Risk score
    hs = df['Hoop stress% of SMYS'] / 100.0
    psp = 1 - ((df['OFF PSP (VE V)'] + 2) / 2)
    max_dist = df['Distance from Pump(KM)'].replace(0, np.nan).max() or 1
    dist_norm = (max_dist - df['Distance from Pump(KM)']) / max_dist
    soil_norm = 1 - np.clip(df['Soil Resistivity (Ω-cm)'] / 10000, 0, 1)
    score = hs * 0.6 + psp * 0.3 + dist_norm * 0.2 + soil_norm * 0.1

    df = pd.concat([df, flags], axis=1)
    df['RiskScore'] = score
    df['FlagsSum'] = flags.sum(axis=1)
    df['RiskCategory'] = df['FlagsSum'].apply(lambda x: 'High' if x >= 4 else ('Medium' if x >= 2 else 'Low'))
    top50 = df.sort_values(['RiskScore', 'Hoop stress% of SMYS', 'OFF PSP (VE V)'], ascending=[False, False, False]).head(50)

    # Save to disk
    df.to_parquet(DATA_CACHE_PATH)
    top50.to_parquet(TOP50_CACHE_PATH)

    st.success("✅ Excel processed and cached. Ready to display!")
    st.experimental_rerun()

# 3️⃣ If no new upload, try to load cached data
elif os.path.exists(DATA_CACHE_PATH) and os.path.exists(TOP50_CACHE_PATH):
    df = pd.read_parquet(DATA_CACHE_PATH)
    top50 = pd.read_parquet(TOP50_CACHE_PATH)
else:
    st.warning("⚠️ Please upload an Excel file to begin.")
    st.stop()

# 4️⃣ UI Plotting and Tables
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

st.subheader("🔥 Top 50 High-Risk Locations")
st.dataframe(top50[['Stationing (m)', 'RiskScore', 'RiskCategory',
                    'Hoop stress% of SMYS', 'OFF PSP (VE V)',
                    'Distance from Pump(KM)', 'Soil Resistivity (Ω-cm)',
                    'Pipe Age', 'CoatingType']], use_container_width=True)

csv = top50.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Top 50 CSV", csv, "Top_50_SCC_Risks.csv", "text/csv")

html_buffer = io.StringIO()
pio.write_html(fig, file=html_buffer, include_plotlyjs='cdn')
st.download_button("⬇️ Download Graph as HTML", html_buffer.getvalue(), "SCC_Graph.html", "text/html")
