import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import io

# -------------------------- PAGE CONFIG --------------------------
st.set_page_config(page_title="📊 SCC Risk Graph Explorer", layout="centered")
st.title("📈 SCC Risk Graph Explorer")

# -------------------------- UPLOAD / LOAD DATA --------------------------
uploaded_file = st.file_uploader("📤 Upload Excel file (.xlsx)", type=["xlsx"])

@st.cache_data(show_spinner=False)
def load_excel(file=None):
    if file is not None:
        df = pd.read_excel(file, engine="openpyxl")
    else:
        url = "https://raw.githubusercontent.com/deepakmahawar150620-beep/VK_Section_SCC/main/Pipeline_VK_Data.xlsx"
        df = pd.read_excel(url, engine="openpyxl")
    df.columns = df.columns.str.strip()
    return df

load_excel.clear()
df = load_excel(uploaded_file)

if uploaded_file:
    st.success("✅ Uploaded file loaded successfully.")
else:
    st.info("ℹ️ Showing default data from GitHub.")

# -------------------------- ENSURE GPS COLUMNS EXIST --------------------------
for col in ["LATITUDE", "LONGITUDE"]:
    if col not in df.columns:
        df[col] = np.nan

# -------------------------- CLEANING / CONVERSIONS --------------------------
if 'OFF PSP (VE V)' in df.columns:
    df['OFF PSP (VE V)'] = pd.to_numeric(df['OFF PSP (VE V)'], errors='coerce').abs()

if 'Hoop stress% of SMYS' in df.columns:
    df['Hoop stress% of SMYS'] = pd.to_numeric(
        df['Hoop stress% of SMYS'].astype(str).str.replace('%', ''),
        errors='coerce'
    )
    if df['Hoop stress% of SMYS'].max(skipna=True) < 10:
        df['Hoop stress% of SMYS'] *= 100

# -------------------------- RISK SCORING FUNCTIONS --------------------------
def scc_risk_score(row):
    score = 0
    try:
        if float(row['Hoop stress% of SMYS']) >= 60:
            score += 10
        if isinstance(row.get('CoatingType'), str) and 'plant cte' in row['CoatingType'].lower():
            score += 10
        if float(row['Distance from Pump(KM)']) < 32:
            score += 10
        if float(row['OFF PSP (VE V)']) > 1.2:
            score += 5
        if float(row['Pipe Age']) > 10:
            score += 10
        if float(row['Temperature']) > 38:
            score += 10
    except:
        pass
    return score

def weighted_risk_score(row):
    try:
        return 0.6 * float(row['Hoop stress% of SMYS']) + \
               0.2 * float(row['Distance from Pump(KM)']) + \
               0.2 * float(row['OFF PSP (VE V)'])
    except:
        return 0.0

# -------------------------- APPLY RISK CALCULATIONS --------------------------
df['SCC Score'] = df.apply(scc_risk_score, axis=1)
df['Weighted Risk Score'] = df.apply(weighted_risk_score, axis=1)
df['SCC Risk Level'] = pd.cut(df['SCC Score'], bins=[-1,19,34,55], labels=['Low','Moderate','High'])

# -------------------------- DISPLAY RISK TABLE --------------------------
st.subheader("📄 SCC Risk Classification Table")
st.dataframe(df, use_container_width=True)
st.download_button("📥 Download Full Risk Data", df.to_csv(index=False), file_name="scc_risk_assessment.csv")

# -------------------------- TOP 50 HIGH‑RISK LOCATIONS --------------------------
top_50 = df[df['SCC Risk Level']=='High'] \
    .sort_values(by='Weighted Risk Score', ascending=False) \
    .head(50)

st.subheader("🔥 Top 50 High‑Risk Locations")
st.dataframe(top_50, use_container_width=True)
st.download_button("⬇️ Download Top 50 High Risk", top_50.to_csv(index=False), file_name="top_50_scc_risks.csv")

# -------------------------- GRAPH PARAMETER SELECT --------------------------
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
selected_col = st.selectbox("📌 Select a parameter to compare with Stationing:", list(plot_columns.keys()))
label = plot_columns[selected_col]

# -------------------------- PLOT GRAPH --------------------------
fig = go.Figure(go.Scatter(
    x=df['Stationing (m)'],
    y=df[selected_col],
    mode='lines+markers',
    name=label,
    line=dict(width=2),
    marker=dict(size=6)
))
if label == 'Hoop Stress (% of SMYS)':
    fig.add_shape(type='line',
                  x0=df['Stationing (m)'].min(),
                  x1=df['Stationing (m)'].max(),
                  y0=60, y1=60,
                  line=dict(color='red', dash='dash'))
elif label == 'OFF PSP (-ve Volt)':
    for y in [0.85,1.2]:
        fig.add_shape(type='line',
                      x0=df['Stationing (m)'].min(),
                      x1=df['Stationing (m)'].max(),
                      y0=y, y1=y,
                      line=dict(color='red', dash='dash'))

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

html_buf = io.StringIO()
pio.write_html(fig, file=html_buf, include_plotlyjs='cdn')
st.download_button("⬇️ Download Graph as HTML",
                   data=html_buf.getvalue(),
                   file_name=f"{label.replace(' ', '_')}_graph.html",
                   mime="text/html")

# -------------------------- DROPDOWN FOR MAP DISPLAY --------------------------
display_option = st.selectbox(
    "🌐 Display top‑50 risk locations on map?",
    options=["No", "Yes – Show Map"],
    index=0
)

if display_option == "Yes – Show Map":
    if {'LATITUDE', 'LONGITUDE'}.issubset(top_50.columns) and not top_50[['LATITUDE','LONGITUDE']].dropna().empty:
        st.subheader("🗺️ Map: Top 50 High‑Risk Locations")
        map_df = top_50.rename(columns={'LATITUDE':'latitude', 'LONGITUDE':'longitude'})
        st.map(map_df[['latitude','longitude']], zoom=10)
    else:
        st.warning("⚠️ GPS data missing — cannot render map.")
