import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import io
from utils.utils import load_data, categorize_risk, get_top_50_risks, generate_pdf_report

st.set_page_config(page_title="SCC Graph Explorer", layout="centered")
st.title("📈 SCC Risk Graph Explorer")

df = load_data()
df_risk, top50 = get_top_50_risks(df)

# Keep all your original graph code (unchanged)
plot_columns = { ... }  # same mapping
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
    fig.add_shape(...)
elif label == 'OFF PSP (-ve Volt)':
    fig.add_shape(...)

fig.update_layout(...)

st.plotly_chart(fig, use_container_width=True)

html_buf = io.StringIO()
pio.write_html(fig, file=html_buf, include_plotlyjs='cdn')
st.download_button("⬇️ Download Graph as HTML", data=html_buf.getvalue(),
                   file_name=f"{label.replace(' ','_')}_graph.html",
                   mime="text/html")

# PDF report download
if st.button("📄 Download PDF Report of Top 50"):
    pdf_bytes = generate_pdf_report(top50, fig)
    st.download_button("⬇️ Download PDF", pdf_bytes, "scc_risk_report.pdf", "application/pdf")
