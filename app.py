import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import io
import os

st.set_page_config(page_title="SCC Risk Explorer", layout="wide")
st.title("📊 Stress Corrosion Cracking (SCC) Risk Dashboard")

# --------------------------
# 🔄 Set up cache file paths
# --------------------------
CACHE_DATA_FILE = "data_cache.parquet"
CACHE_TOP50_FILE = "top50_cache.parquet"

# --------------------------
# 📤 Permanent uploader
# --------------------------
uploaded_file = st.file_uploader("📂 Upload Excel file", type=["xlsx"], help="Upload your pipeline data Excel file")

# --------------------------
# 🧠 Session & File Cache Logic
# --------------------------
def process_excel(df):
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

    flags = pd.DataFrame({
        'Stress>60': (df['Hoop stress% of SMYS'] > 60).astype(int),
        'Age>10yrs': (df['Pipe Age'] > 10).astype(int),
        'Temp>38C': (df['Temperature'] > 38).astype(int),
        'Dist≤32km': (df['Distance from Pump(KM)'] <= 32).astype(int),
        'CoatingHighRisk': (~df['CoatingType'].str.upper().isin(['FBE', 'LIQUID EPOXY'])).astype(int),
        'Soil<5000': (df['Soil Resistivity (Ω-cm)'] < 5000).astype(int),
        'OFFPSP>−1.2V': (df['OFF PSP (VE V)'] > -1.2).astype(int)
    })

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

    return df, top50

# --------------------------
# ⏫ Load from upload or cache
# --------------------------
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    full_df, top50_df = process_excel(df)

    full_df.to_parquet(CACHE_DATA_FILE)
    top50_df.to_parquet(CACHE_TOP50_FILE)

    st.session_state["data"] = full_df
    st.session_state["top50"] = top50_df
    st.success("✅ File uploaded and processed.")
    st.experimental_rerun()

elif "data" in st.session_state and "top50" in st.session_state:
    full_df = st.session_state["data"]
    top50_df = st.session_state["top50"]

elif os.path.exists(CACHE_DATA_FILE) and os.path.exists(CACHE_TOP50_FILE):
    full_df = pd.read_parquet(CACHE_DATA_FILE)
    top50_df = pd.read_parquet(CACHE_TOP50_FILE)
    st.session_state["data"] = full_df
    st.session_state["top50"] = top50_df
else:
    st.warning("🟡 Please upload an Excel file to begin.")
    st.stop()

# --------------------------
# 📊 Graph (Revised)
# --------------------------
plot_columns = {
    'Depth (mm)': 'Depth (mm)',
    'OFF PSP (VE V)': 'OFF PSP (-ve Volt)',
    'Soil Resistivity (Ω-cm)': 'Soil Resistivity (Ω-cm)',
    'Distance from Pump(KM)': 'Distance from Pump (KM)',
    'Operating Pr.': 'Operating Pressure',
    'Remaining Thickness(mm)': 'Remaining Thickness (mm)',
    'Hoop stress% of SMYS': 'Hoop Stress (% of SMYS)',
    'Temperature': 'Temperature (°C)',
    'Pipe Age': 'Pipe Age'
}

selected_col = st.selectbox("Select a parameter to compare with Stationing:", list(plot_columns.keys()))
label = plot_columns[selected_col]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=full_df['Stationing (m)'],
    y=full_df[selected_col],
    mode='lines+markers',
    name=label,
    line=dict(width=2),
    marker=dict(size=6)
))

if label == 'Hoop Stress (% of SMYS)':
    fig.add_shape(type='line', x0=full_df['Stationing (m)'].min(), x1=full_df['Stationing (m)'].max(),
                  y0=60, y1=60, line=dict(color='red', dash='dash'))
elif label == 'OFF PSP (-ve Volt)':
    for yval in [0.85, 1.2]:
        fig.add_shape(type='line', x0=full_df['Stationing (m)'].min(), x1=full_df['Stationing (m)'].max(),
                      y0=yval, y1=yval, line=dict(color='red', dash='dash'))

fig.update_layout(
    title=f"Stationing vs {label}",
    xaxis_title="Stationing (m)",
    yaxis_title=label,
    height=500,
    template='plotly_white',
    xaxis=dict(showline=True, linecolor='black', mirror=True),
    yaxis=dict(showline=True, linecolor='black', mirror=True, gridcolor='lightgray'),
    margin=dict(l=60, r=40, t=50, b=60)
)

st.plotly_chart(fig, use_container_width=True)

html_buffer = io.StringIO()
pio.write_html(fig, file=html_buffer, include_plotlyjs='cdn')
st.download_button("⬇️ Download High-Quality Graph as HTML", html_buffer.getvalue(), f"{label.replace(' ', '_')}_graph.html", "text/html")

# --------------------------
# 📄 Table Section
# --------------------------
st.subheader("🔥 Top 50 High-Risk Locations")
st.dataframe(top50_df[['Stationing (m)', 'RiskScore', 'RiskCategory',
                       'Hoop stress% of SMYS', 'OFF PSP (VE V)',
                       'Distance from Pump(KM)', 'Soil Resistivity (Ω-cm)',
                       'Pipe Age', 'CoatingType']], use_container_width=True)

# --------------------------
# 📥 Downloads
# --------------------------
csv = top50_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Top 50 CSV", csv, "Top_50_SCC_Risks.csv", "text/csv")
