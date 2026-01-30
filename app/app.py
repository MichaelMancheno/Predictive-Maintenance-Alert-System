import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import sys
import numpy as np
from pathlib import Path

# Agregar path
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="NASA Bearing Predictor", page_icon="⚙️", layout="wide")

st.title("⚙️ NASA Bearing Failure Prediction System")

# Cargar modelos
@st.cache_resource
def load_models():
    model_path = Path(__file__).parent.parent / 'models' / 'model.pkl'
    scaler_path = Path(__file__).parent.parent / 'models' / 'scaler.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_models()

# --- SIDEBAR: RANGOS AMPLIADOS PARA CUBRIR ISO 10816 ZONA D ---
st.sidebar.header("📊 Vibration Parameters")

# Ajuste: Rangos ampliados (antes 2.0, ahora 50.0 para simular fallas catastróficas)
max_vib = st.sidebar.slider("Max Vibration (g)", 0.0, 10.0, 0.5, 0.1)
mean_vib = st.sidebar.slider("Mean Vibration", 0.0, 5.0, 0.3, 0.1)
std_vib = st.sidebar.slider("Std Vibration", 0.0, 5.0, 0.1, 0.1)

# Ajuste: RMS ampliado hasta 15 mm/s (La zona D empieza en 7.1)
rms_vib = st.sidebar.slider("RMS Vibration (mm/s)", 0.0, 15.0, 0.4, 0.1)

hours = st.sidebar.number_input("Hours Operation", 0, 20000, 5000)

# --- CÁLCULO DE VARIABLES DE INGENIERÍA (Backend) ---
# Calculamos las relaciones en vivo para que el modelo reaccione a tu slider
rms_diff = 0.05  # Valor base estable
severity_ratio = max_vib / (rms_vib + 1e-6)
relative_std = std_vib / (mean_vib + 1e-6)

# Crear DataFrame para el modelo
features = pd.DataFrame([[
    max_vib, mean_vib, std_vib, rms_vib, hours, 
    rms_diff, severity_ratio, relative_std
]], columns=[
    'max_vibration', 'mean_vibration', 'std_vibration', 
    'rms_vibration', 'hours_operation', 'rms_diff', 
    'severity_ratio', 'relative_std'
])

# Predicción del Modelo
features_scaled = scaler.transform(features)
probability = model.predict_proba(features_scaled)[0]

# --- LÓGICA HÍBRIDA (SOLUCIÓN AL "SALTO" BRUSCO) ---
# El modelo de ML a veces es muy binario. Aquí combinamos su predicción
# con la realidad física de la ISO 10816 para suavizar la gráfica.

risk_ml = probability[1] * 100 if len(probability) > 1 else 0

# Factor ISO: Si el RMS es alto, el riesgo DEBE ser alto (física pura)
if rms_vib > 7.1:     # Zona D
    iso_risk = 95.0
elif rms_vib > 4.5:   # Zona C alta
    iso_risk = 75.0
elif rms_vib > 2.8:   # Zona C baja
    iso_risk = 50.0
else:
    iso_risk = 10.0

# El riesgo final es un promedio ponderado: 
# 60% lo que dice el Modelo ML (datos NASA) + 40% lo que dice la ISO (Física)
# Esto evita que salte de 20 a 90 solo por cambiar la hora.
final_risk = (risk_ml * 0.6) + (iso_risk * 0.4)

# Asegurar límites
final_risk = min(max(final_risk, 0), 100)

# Determinación de Estado
if final_risk > 70:
    status_text = "⚠️ FAILURE IMMINENT"
    status_color = "error"
elif final_risk > 40:
    status_text = "⚠️ WARNING"
    status_color = "warning"
else:
    status_text = "✅ NORMAL"
    status_color = "success"

# --- VISUALIZACIÓN ---
col1, col2, col3 = st.columns(3)

with col1:
    if status_color == "error":
        st.error(status_text)
    elif status_color == "warning":
        st.warning(status_text)
    else:
        st.success(status_text)
    
with col2:
    st.metric("Failure Risk (Hybrid)", f"{final_risk:.1f}%")

with col3:
    # Lógica ISO 10816 (Clase II - Máquinas Medianas)
    if rms_vib < 1.12:
        iso = "A - Good"
        iso_color = "green"
    elif rms_vib < 2.80:
        iso = "B - Acceptable"  
        iso_color = "blue"
    elif rms_vib < 7.10:
        iso = "C - Unsatisfactory"
        iso_color = "orange"
    else:
        iso = "D - Unacceptable"
        iso_color = "red"
    st.markdown(f"**ISO 10816:** <span style='color:{iso_color}'>**{iso}**</span>", unsafe_allow_html=True)

# Gauge Chart
st.subheader("📈 Operational Risk Assessment")
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = final_risk,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Failure Probability (%)"},
    gauge = {
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkred" if final_risk > 75 else "orange" if final_risk > 40 else "green"},
        'steps': [
            {'range': [0, 40], 'color': 'lightgreen'},
            {'range': [40, 75], 'color': 'lightyellow'},
            {'range': [75, 100], 'color': 'salmon'}
        ],
        'threshold': {'line': {'color': "red", 'width': 4}, 'value': 80}
    }
))
st.plotly_chart(fig, use_container_width=True)

# Recomendaciones
st.subheader("💡 Engineering Recommendations")
recs = []
if final_risk > 75:
    recs = [
        "🔴 **CRITICAL:** Machine vibration in Zone D. Immediate shutdown recommended.",
        "🔍 Check for bearing seizure or severe misalignment.",
        "📋 Schedule full replacement."
    ]
elif final_risk > 40:
    recs = [
        "🟠 **WARNING:** Vibration increasing (Zone C). Plan maintenance soon.",
        "📅 Reduce inspection interval to daily.",
        "👀 Monitor temperature trends."
    ]
else:
    recs = [
        "🟢 **NORMAL:** Operation within safe limits (Zone A/B).",
        "📅 Maintain regular schedule.",
        "✅ No action required."
    ]

for rec in recs:
    st.markdown(f"• {rec}")

st.markdown("---")
st.markdown("**Developed by Michael Mancheno** | Industrial Engineer + ML Specialist")