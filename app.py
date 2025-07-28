import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import io

# ---------- Data Functions ----------
def load_data():
    df = pd.read_excel("https://raw.githubusercontent.com/deepakmahawar150620-beep/SCC_Pawan/main/Pipeline_data.xlsx", engine="openpyxl")
    df.columns = df.columns.str.strip()
    df['OFF PSP (VE V)'] = pd.to_numeric(df['OFF PSP (VE V)'], errors='coerce').abs().fillna(0)
    hs = pd.to_numeric(df['Hoop stress% of SMYS'].astype(str).str.replace('%',''), errors='coerce').fillna(0)
    if hs.max() < 10: hs *= 100
    df['Hoop stress% of SMYS'] = hs
    df['Distance from Pump(KM)'] = pd.to_numeric(df.get('Distance from Pump(KM)',0), errors='coerce').fillna(1e6)
    df['Pipe Age'] = pd.to_numeric(df.get('Pipe Age',0), errors='coerce').fillna(0)
    df['Temperature'] = pd.to_numeric(df.get('Temperature',0), errors='coerce').fillna(0)
    df['Soil Resistivity (Ω-cm)'] = pd.to_numeric(df.get('Soil Resistivity (Ω-cm)',0), errors='coerce').fillna(1e9)
    df['CoatingType'] = df.get('CoatingType','').astype(str)
    return df

def flag_criteria(df):
    return pd.DataFrame({
        'Stress>60': (df['Hoop stress% of SMYS'] > 60).astype(int),
        'Age>10yrs': (df['Pipe Age'] > 10).astype(int),
        'Temp>38C': (df['Temperature'] > 38).astype(int),
        'Dist≤32km': (df['Distance from Pump(KM)'] <= 32).astype(int),
        'CoatingHighRisk': (~df['CoatingType'].str.upper().isin(['FBE','LIQUID EPOXY'])).astype(int),
        'Soil<5000': (df['Soil Resistivity (Ω-cm)'] < 5000).astype(int),
        'OFFPSP>−1.2V': (df['OFF PSP (VE V)'] > -1.2).astype(int)
    })

def compute_risk_score(df):
    flags = flag_criteria(df)
    hs = df['Hoop stress% of SMYS']/100.0
    psp = 1 - ((df['OFF PSP (VE V)'] + 2)/2)
    max_dist = df['Distance from Pump(KM)'].replace(0, np.nan).max() or 1
    dist_norm = (max_dist - df['Distance from Pump(KM)'])/max_dist
    soil_norm = 1 - np.clip(df['Soil Resistivity (Ω-cm)']/10000, 0,1)
    w = {'hs':0.6, 'psp':0.3, 'dist':0.2, 'soil':0.1}
    return hs*w['hs'] + psp*w['psp'] + dist_norm*w['dist'] + soil_norm*w['soil'], flags

def get_top_50_risks(df):
    dfc = df.copy()
    risk_score, flags = compute_risk_score(dfc)
    dfc = pd.concat([dfc, flags], axis=1)
    dfc['RiskScore'] = risk_score
    dfc['FlagsSum'] = flags.sum(axis=1)
    dfc['RiskCategory'] = dfc['FlagsSum'].apply(lambda x: 'High' if x>=4 else ('Medium' if x>=2 else 'Low'))
    top50 = dfc.sort_values(['RiskScore','Hoop stress% of SMYS','OFF PSP (VE V)'],
                            ascending=[False, False, False]).head(50)
    return dfc, top50

# ---------- Streamlit App ----------
st.set_page_config(page_title="SCC Graph Explorer", layout="centered")
st.title("📈 SCC Risk Graph Explorer")

df = load_data()
df_risk, top50 = get_top_50_risks(df)

plot_columns = {
    'Hoop stress% of SMYS': 'Hoop Stress (% of SMYS)',
    'OFF PSP (VE V)': 'OFF PSP (-ve Volt)',
    'Distance from Pump(KM)': 'Distance from Pump (km)',
    'Soil Resistivity (Ω-cm)': 'Soil Resistivity (Ω-cm)',
    'Pipe Age': 'Pipe Age (years)',
    'Temperature': 'Temperature (°C)'
}

selected_col = st.selectbox("Select a parameter...", list(plot_columns.keys()))
label = plot_columns[selected_col]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['Stationing (m)'],
    y=df[selected_col],
    mode='lines+markers',
    name=label,
    line=dict(width=2),
    marker=dict(size=6)
))

if label == 'Hoop Stress (% of SMYS)':
    fig.add_shape(type="line", x0=min(df['Stationing (m)']), x1=max(df['Stationing (m)']),
                  y0=60, y1=60, line=dict(color="red", width=2, dash="dash"))
elif label == 'OFF PSP (-ve Volt)':
    fig.add_shape(type="line", x0=min(df['Stationing (m)']), x1=max(df['Stationing (m)']),
                  y0=-0.85, y1=-0.85, line=dict(color="green", width=2, dash="dot"))

fig.update_layout(title=label + " vs Stationing",
                  xaxis_title="Stationing (m)",
                  yaxis_title=label,
                  template="plotly_white")

st.plotly_chart(fig, use_container_width=True)

# HTML graph download
html_buf = io.StringIO()
pio.write_html(fig, file=html_buf, include_plotlyjs='cdn')
st.download_button("⬇️ Download Graph as HTML", data=html_buf.getvalue(),
                   file_name=f"{label.replace(' ','_')}_graph.html", mime="text/html")
