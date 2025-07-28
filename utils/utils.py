# utils/utils.py
import pandas as pd
import numpy as np
import io

def load_data():
    df = pd.read_excel("https://raw.githubusercontent.com/deepakmahawar150620-beep/SCC_Pawan/main/Pipeline_data.xlsx", engine="openpyxl")
    df.columns = df.columns.str.strip()
    # clean and convert key columns:
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

def generate_pdf_report(top50_df, fig):
    # uses reportlab and fig.to_image to build PDF similar to prior code
    ...  # same as earlier PDF snippet
