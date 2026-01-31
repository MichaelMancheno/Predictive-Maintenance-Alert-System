"""
Predictive Maintenance Dashboard - RUL Prediction
Based on ISO 10816-1 Group 2 and NASA bearing data
"""
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path

# ===== CONFIGURACIÓN =====
st.set_page_config(
    page_title="RUL Predictor | ISO 10816",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar cerrado por defecto
)

# ===== CARGAR MODELOS =====
@st.cache_resource
def load_models():
    """Cargar modelo y scaler desde disco"""
    base_path = Path(__file__).parent.parent
    model_path = base_path / 'models' / 'model.pkl'
    scaler_path = base_path / 'models' / 'scaler.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_models()
except Exception as e:
    st.error(f"❌ Error cargando modelos: {e}")
    st.info("💡 Asegúrate de haber ejecutado: python src/train_model.py")
    st.stop()

# ===== FUNCIONES AUXILIARES =====
def classify_iso10816_group2(rms_mms):
    """
    Clasificación ISO 10816-1 Grupo 2
    Máquinas medianas (15-75 kW) en fundaciones rígidas
    """
    if rms_mms < 2.8:
        return 'A', 'Excelente', 'green'
    elif rms_mms < 4.5:
        return 'B', 'Aceptable', 'lightgreen'
    elif rms_mms < 7.1:
        return 'C', 'Tolerable', 'orange'
    else:
        return 'D', 'Inaceptable', 'red'

# ===== HEADER =====
st.title("⚙️ Sistema de Mantenimiento Predictivo")
st.markdown("**Predicción de Vida Útil Remanente (RUL) | Cumplimiento ISO 10816-1 Grupo 2**")
st.markdown("---")

# ===== FORMULARIO DE ENTRADA (LAYOUT HORIZONTAL) =====
st.subheader("📝 Ingrese los Parámetros del Equipo")

# Crear columnas para inputs
col_input1, col_input2, col_input3 = st.columns(3)

with col_input1:
    st.markdown("#### 🔹 Métricas de Vibración Básicas")
    
    rms_vib = st.number_input(
        "**RMS Vibración** (mm/s)", 
        min_value=0.0, 
        max_value=20.0, 
        value=3.5, 
        step=0.1,
        help="Métrica principal ISO 10816. Límites Grupo 2: A<2.8, B<4.5, C<7.1, D>7.1",
        key="rms"
    )
    
    max_vib = st.number_input(
        "**Vibración Máxima** (mm/s)", 
        min_value=0.0, 
        max_value=30.0, 
        value=5.0, 
        step=0.1,
        key="max"
    )
    
    mean_vib = st.number_input(
        "**Vibración Media** (mm/s)", 
        min_value=0.0, 
        max_value=15.0, 
        value=2.5, 
        step=0.1,
        key="mean"
    )

with col_input2:
    st.markdown("#### 🔹 Métricas de Vibración Avanzadas")
    
    std_vib = st.number_input(
        "**Desviación Estándar** (mm/s)", 
        min_value=0.0, 
        max_value=10.0, 
        value=1.2, 
        step=0.1,
        key="std"
    )
    
    peak_to_peak = st.number_input(
        "**Pico a Pico** (mm/s)", 
        min_value=0.0, 
        max_value=50.0, 
        value=8.0, 
        step=0.5,
        key="p2p"
    )
    
    hours_operation = st.number_input(
        "**Horas de Operación** (h)", 
        min_value=0, 
        max_value=50000, 
        value=5000, 
        step=100,
        key="hours"
    )

with col_input3:
    st.markdown("#### 🔹 Indicadores de Diagnóstico")
    
    kurtosis = st.number_input(
        "**Kurtosis** (adimensional)", 
        min_value=-5.0, 
        max_value=20.0, 
        value=3.0, 
        step=0.1,
        help="Kurtosis > 3 indica comportamiento impulsivo (fallas en rodamientos)",
        key="kurt"
    )
    
    crest_factor = st.number_input(
        "**Factor de Cresta** (adimensional)", 
        min_value=1.0, 
        max_value=10.0, 
        value=3.5, 
        step=0.1,
        help="Relación Pico/RMS. Valores altos indican impactos",
        key="crest"
    )
    
    # Espaciador
    st.markdown("")
    st.markdown("")

# ===== BOTÓN DE CÁLCULO (CENTRADO Y GRANDE) =====
st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    calculate_button = st.button(
        "🚀 CALCULAR RUL", 
        type="primary",
        use_container_width=True,
        help="Haz clic para calcular la Vida Útil Remanente"
    )

st.markdown("---")

# ===== RESULTADOS (SOLO SE MUESTRAN DESPUÉS DE HACER CLIC) =====
if calculate_button:
    
    # Preparar features
    features = pd.DataFrame([[
        max_vib, mean_vib, std_vib, rms_vib, 
        peak_to_peak, kurtosis, crest_factor, hours_operation
    ]], columns=[
        'max_vibration', 'mean_vibration', 'std_vibration', 'rms_vibration',
        'peak_to_peak', 'kurtosis', 'crest_factor', 'hours_operation'

    ])
    
    # Escalar y predecir
    features_scaled = scaler.transform(features)
    predicted_rul = model.predict(features_scaled)[0]
    
    # Clasificación ISO
    iso_class, iso_desc, iso_color = classify_iso10816_group2(rms_vib)
    
    # ===== RESULTADOS PRINCIPALES (UNA SOLA FILA) =====
    st.subheader("📊 Resultados del Análisis")
    
    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
    
    with result_col1:
        # RUL en horas
        st.metric(
            "🕐 Vida Útil Remanente (RUL)", 
            f"{predicted_rul:.1f} horas",
            delta=f"{(predicted_rul/24):.1f} días"
        )
    
    with result_col2:
        # RUL en días (más fácil de entender)
        days_remaining = predicted_rul / 24
        st.metric(
            "📅 Días Hasta Falla",
            f"{days_remaining:.1f} días",
            delta=f"{(days_remaining/7):.1f} semanas" if days_remaining > 7 else None
        )
    
    with result_col3:
        # Estado del equipo
        if predicted_rul < 24:
            st.error("🚨 **CRÍTICO**")
            status = "Detener Ahora"
            status_color = "red"
        elif predicted_rul < 168:  # 1 semana
            st.warning("⚠️ **PRECAUCIÓN**")
            status = "Planear Mantenimiento"
            status_color = "orange"
        else:
            st.success("✅ **SALUDABLE**")
            status = "Operación Normal"
            status_color = "green"
        
        st.metric("Estado del Equipo", status)
    
    with result_col4:
        # Clasificación ISO
        if iso_class == 'A':
            st.success(f"**Clase ISO: {iso_class}**")
        elif iso_class == 'B':
            st.info(f"**Clase ISO: {iso_class}**")
        elif iso_class == 'C':
            st.warning(f"**Clase ISO: {iso_class}**")
        else:
            st.error(f"**Clase ISO: {iso_class}**")
        
        st.metric("Condición ISO 10816", iso_desc)
    
    st.markdown("---")
    
    # ===== VISUALIZACIONES =====
    viz_col1, viz_col2 = st.columns(2)
    
    # Gauge de RUL
    with viz_col1:
        st.subheader("📈 Porcentaje de Vida Remanente")
        
        # Convertir RUL a porcentaje
        max_expected_life = 200  # horas (ajusta según dataset)
        rul_percentage = min((predicted_rul / max_expected_life) * 100, 100)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rul_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Vida Remanente (%)", 'font': {'size': 24}},
            number={'suffix': "%", 'font': {'size': 48}},
            delta={'reference': 50, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkgray"},
                'bar': {'color': "green" if rul_percentage > 50 else "orange" if rul_percentage > 20 else "red", 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 3,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(255,0,0,0.3)'},
                    {'range': [20, 50], 'color': 'rgba(255,165,0,0.3)'},
                    {'range': [50, 100], 'color': 'rgba(0,255,0,0.3)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 6},
                    'thickness': 0.8,
                    'value': 20
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=80, b=20),
            font={'size': 16, 'family': 'Arial'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Indicador numérico
        if predicted_rul < 24:
            st.error(f"⚠️ **¡URGENTE!** Solo quedan **{predicted_rul:.1f} horas** ({days_remaining:.1f} días)")
        elif predicted_rul < 168:
            st.warning(f"⏰ Quedan **{predicted_rul:.1f} horas** ({days_remaining:.1f} días) - Programar mantenimiento")
        else:
            st.success(f"✅ Quedan **{predicted_rul:.1f} horas** ({days_remaining:.1f} días) de operación segura")
    
    # Zonas ISO 10816
    with viz_col2:
        st.subheader("🎯 Clasificación ISO 10816-1 Grupo 2")
        
        # Gráfico de zonas ISO
        fig_iso = go.Figure()
        
        # Zonas ISO
        zones = [
            {'name': 'Zona A - Excelente', 'min': 0, 'max': 2.8, 'color': 'rgba(0,255,0,0.3)'},
            {'name': 'Zona B - Aceptable', 'min': 2.8, 'max': 4.5, 'color': 'rgba(144,238,144,0.5)'},
            {'name': 'Zona C - Tolerable', 'min': 4.5, 'max': 7.1, 'color': 'rgba(255,165,0,0.5)'},
            {'name': 'Zona D - Inaceptable', 'min': 7.1, 'max': 15, 'color': 'rgba(255,0,0,0.3)'}
        ]
        
        # Crear barras horizontales para cada zona
        for idx, zone in enumerate(zones):
            fig_iso.add_trace(go.Bar(
                y=[zone['name']],
                x=[zone['max'] - zone['min']],
                base=[zone['min']],
                orientation='h',
                marker=dict(color=zone['color'], line=dict(color='gray', width=1)),
                name=zone['name'],
                text=f"{zone['min']}-{zone['max']} mm/s",
                textposition='inside',
                hovertemplate=f"<b>{zone['name']}</b><br>Rango: {zone['min']}-{zone['max']} mm/s<extra></extra>"
            ))
        
        # Marcador de valor actual (línea vertical)
        fig_iso.add_vline(
            x=rms_vib, 
            line_width=4, 
            line_dash="dash", 
            line_color="darkblue",
            annotation_text=f"Actual: {rms_vib:.2f} mm/s",
            annotation_position="top"
        )
        
        # Marcador de valor actual (punto)
        current_zone = next((z['name'] for z in zones if z['min'] <= rms_vib < z['max']), zones[-1]['name'])
        
        fig_iso.add_trace(go.Scatter(
            x=[rms_vib],
            y=[current_zone],
            mode='markers',
            marker=dict(size=25, color='darkblue', symbol='diamond', line=dict(color='white', width=2)),
            name='Valor Actual',
            hovertemplate=f'<b>RMS Actual</b><br>{rms_vib:.2f} mm/s<br>Zona {iso_class}<extra></extra>',
            showlegend=False
        ))
        
        fig_iso.update_layout(
            title=f"Nivel Actual de Vibración: Zona {iso_class}",
            xaxis_title="Velocidad RMS (mm/s)",
            yaxis_title="",
            height=400,
            margin=dict(l=20, r=20, t=80, b=50),
            showlegend=False,
            xaxis=dict(range=[0, 15], dtick=1),
            barmode='overlay',
            font={'size': 14}
        )
        
        st.plotly_chart(fig_iso, use_container_width=True)
        
        # Descripción de la zona actual
        if iso_class == 'A':
            st.success(f"✅ **Zona {iso_class}**: Condición excelente - Equipo recién comisionado o en perfecto estado")
        elif iso_class == 'B':
            st.info(f"ℹ️ **Zona {iso_class}**: Condición aceptable - Operación sin restricciones a largo plazo")
        elif iso_class == 'C':
            st.warning(f"⚠️ **Zona {iso_class}**: Condición tolerable - Operar por tiempo limitado, programar mantenimiento")
        else:
            st.error(f"🚨 **Zona {iso_class}**: Condición inaceptable - Acción inmediata requerida")
    
    # ===== ANÁLISIS DE MÉTRICAS =====
    st.markdown("---")
    st.subheader("📊 Análisis Detallado de Parámetros")
    
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        # Gráfico de barras de métricas de vibración
        metrics_data = pd.DataFrame({
            'Métrica': ['Máxima', 'RMS', 'Media', 'Desv. Std', 'Pico-Pico'],
            'Valor (mm/s)': [max_vib, rms_vib, mean_vib, std_vib, peak_to_peak]
        })
        
        fig_metrics = px.bar(
            metrics_data,
            x='Métrica',
            y='Valor (mm/s)',
            title="Comparación de Parámetros de Vibración",
            color='Valor (mm/s)',
            color_continuous_scale='Reds',
            text='Valor (mm/s)'
        )
        
        fig_metrics.update_traces(
            texttemplate='%{text:.2f}', 
            textposition='outside',
            textfont_size=14
        )
        fig_metrics.update_layout(
            height=350, 
            showlegend=False,
            font={'size': 14}
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with analysis_col2:
        # Indicadores de diagnóstico avanzado
        st.markdown("#### 🔬 Diagnósticos Avanzados")
        
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            # Kurtosis
            kurtosis_status = "Normal" if kurtosis < 4 else "Advertencia" if kurtosis < 6 else "Crítico"
            kurtosis_color = "green" if kurtosis < 4 else "orange" if kurtosis < 6 else "red"
            
            fig_kurt = go.Figure(go.Indicator(
                mode="gauge+number",
                value=kurtosis,
                title={'text': "Kurtosis", 'font': {'size': 18}},
                number={'font': {'size': 32}},
                gauge={
                    'axis': {'range': [0, 10]},
                    'bar': {'color': kurtosis_color, 'thickness': 0.7},
                    'steps': [
                        {'range': [0, 4], 'color': 'rgba(0,255,0,0.2)'},
                        {'range': [4, 6], 'color': 'rgba(255,255,0,0.2)'},
                        {'range': [6, 10], 'color': 'rgba(255,0,0,0.2)'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'value': 6}
                }
            ))
            fig_kurt.update_layout(height=220, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_kurt, use_container_width=True)
            
            if kurtosis < 4:
                st.success(f"✅ **{kurtosis_status}**")
            elif kurtosis < 6:
                st.warning(f"⚠️ **{kurtosis_status}**")
            else:
                st.error(f"🚨 **{kurtosis_status}**")
            
            st.caption("Normal: <4 | Impulsivo: >6")
        
        with diag_col2:
            # Crest Factor
            cf_status = "Normal" if crest_factor < 4 else "Advertencia" if crest_factor < 6 else "Crítico"
            cf_color = "green" if crest_factor < 4 else "orange" if crest_factor < 6 else "red"
            
            fig_cf = go.Figure(go.Indicator(
                mode="gauge+number",
                value=crest_factor,
                title={'text': "Factor de Cresta", 'font': {'size': 18}},
                number={'font': {'size': 32}},
                gauge={
                    'axis': {'range': [1, 10]},
                    'bar': {'color': cf_color, 'thickness': 0.7},
                    'steps': [
                        {'range': [1, 4], 'color': 'rgba(0,255,0,0.2)'},
                        {'range': [4, 6], 'color': 'rgba(255,255,0,0.2)'},
                        {'range': [6, 10], 'color': 'rgba(255,0,0,0.2)'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'value': 6}
                }
            ))
            fig_cf.update_layout(height=220, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_cf, use_container_width=True)
            
            if crest_factor < 4:
                st.success(f"✅ **{cf_status}**")
            elif crest_factor < 6:
                st.warning(f"⚠️ **{cf_status}**")
            else:
                st.error(f"🚨 **{cf_status}**")
            
            st.caption("Normal: <4 | Impactos: >6")
    
    # ===== RECOMENDACIONES =====
    st.markdown("---")
    st.subheader("💡 Recomendaciones de Mantenimiento")
    
    if predicted_rul < 24:
        st.error("### 🚨 CRÍTICO - ACCIÓN INMEDIATA REQUERIDA")
        recommendations = [
            f"🔴 **DETENER OPERACIÓN INMEDIATAMENTE** - Solo quedan {predicted_rul:.1f} horas",
            "📞 **Alerta de mantenimiento de emergencia** - Contactar equipo de guardia",
            "🔧 **Preparar reemplazo de rodamiento** - Ordenar repuestos AHORA",
            "📋 **Documentar condiciones actuales** - Niveles de vibración, temperatura, ruido",
            "⚠️ **Aislar equipo** - Prevenir fallas en cascada"
        ]
        
    elif predicted_rul < 168:
        st.warning("### ⚠️ PRECAUCIÓN - PROGRAMAR MANTENIMIENTO URGENTE")
        recommendations = [
            f"🟡 **Programar mantenimiento dentro de {predicted_rul/24:.1f} días** - Coordinar con producción",
            "📦 **Ordenar repuestos** - Rodamiento, sellos, lubricante",
            "📈 **Aumentar monitoreo** - Verificar cada 4-8 horas",
            "📊 **Análisis de tendencias** - Monitorear tasa de aumento de vibración",
            "🔍 **Inspeccionar componentes relacionados** - Acoplamiento, alineación, lubricación"
        ]
        
    else:
        st.success("### ✅ OPERACIÓN NORMAL - MONITOREO DE RUTINA")
        recommendations = [
            f"🟢 **Operación segura por {predicted_rul/24:.1f} días** - Continuar programa normal",
            "📅 **Planear mantenimiento** durante próxima parada programada",
            "📊 **Monitoreo semanal** - Seguir tendencias de vibración",
            "🔧 **Tareas preventivas** - Lubricación, verificaciones de alineación",
            "📝 **Actualizar registro de mantenimiento** - Registrar lecturas actuales"
        ]
    
    # Mostrar recomendaciones
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Recomendaciones ISO
    st.markdown("#### 📋 Guías ISO 10816:")
    
    if iso_class == 'A':
        st.success("✅ **Zona A**: Equipo recién comisionado o en excelente condición - Continuar operación")
    elif iso_class == 'B':
        st.info("ℹ️ **Zona B**: Aceptable para operación sin restricciones a largo plazo")
    elif iso_class == 'C':
        st.warning("⚠️ **Zona C**: Tolerable por períodos limitados - Programar mantenimiento pronto")
    else:
        st.error("🚨 **Zona D**: Inaceptable - Acción inmediata requerida para prevenir daños")

else:
    # Mensaje cuando no se ha calculado aún
    st.info("👆 **Ingrese los parámetros del equipo arriba y haga clic en 'CALCULAR RUL' para ver los resultados**")
    
    # Mostrar información útil mientras tanto
    st.markdown("---")
    st.subheader("📚 Información del Sistema")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        ### 🎯 ¿Qué es RUL?
        
        **RUL (Remaining Useful Life)** es el tiempo estimado hasta que el equipo 
        requiera mantenimiento o falle.
        
        **Ventajas de predecir RUL:**
        - ✅ Planificación de mantenimiento sin interrupciones inesperadas
        - ✅ Reducción de costos de downtime
        - ✅ Optimización de inventario de repuestos
        - ✅ Mayor seguridad operacional
        """)
    
    with info_col2:
        st.markdown("""
        ### 📏 ISO 10816-1 Grupo 2
        
        **Aplicable a:**  
        Máquinas medianas (15-75 kW) en fundaciones rígidas
        
        **Zonas de Severidad:**
        
        | Zona | RMS (mm/s) | Estado |
        |------|-----------|--------|
        | A | < 2.8 | Excelente |
        | B | 2.8 - 4.5 | Aceptable |
        | C | 4.5 - 7.1 | Tolerable |
        | D | > 7.1 | Inaceptable |
        """)

# ===== INFORMACIÓN TÉCNICA (SIEMPRE VISIBLE) =====
st.markdown("---")
with st.expander("🔬 Información Técnica y Detalles del Modelo"):
    
    tab1, tab2, tab3 = st.tabs(["Modelo ML", "Norma ISO 10816", "Definición de Features"])
    
    with tab1:
        st.markdown("""
        ### 🤖 Modelo de Machine Learning
        
        **Algoritmo:** Random Forest Regressor  
        **Datos de Entrenamiento:** NASA IMS Bearing Dataset  
        **Features:** 8 métricas de vibración ingenierizadas  
        **Target:** Vida Útil Remanente (RUL) en horas  
        **Rendimiento:** MAE ≈ ±15 horas, R² > 0.90  
        
        **Enfoque de Predicción:**
        - Analiza patrones de vibración multidimensionales
        - Identifica tendencias de degradación específicas de fallas en rodamientos
        - Proporciona estimación continua de RUL (no solo saludable/fallido binario)
        
        **Por qué se necesita ML:**
        - ISO 10816 proporciona clasificación de severidad pero NO tiempo hasta falla
        - La predicción de RUL requiere reconocimiento de patrones en múltiples features
        - Detección temprana de tendencias sutiles de degradación antes de alcanzar límites ISO
        """)
    
    with tab2:
        st.markdown("""
        ### 📏 Norma ISO 10816-1
        
        **Clasificación Grupo 2:**  
        Máquinas medianas (15-75 kW) en fundaciones rígidas
        
        | Zona | Rango RMS (mm/s) | Descripción | Acción |
        |------|------------------|-------------|--------|
        | **A** | < 2.8 | Excelente | Recién comisionado o excelente condición |
        | **B** | 2.8 - 4.5 | Aceptable | Operación sin restricciones a largo plazo |
        | **C** | 4.5 - 7.1 | Tolerable | Operación limitada, programar mantenimiento |
        | **D** | > 7.1 | Inaceptable | Acción inmediata requerida |
        
        **Medición:**
        - Velocidad RMS de vibración
        - Medida en carcasas de rodamientos
        - Rango de frecuencia: 10-1000 Hz
        - Dirección: Radial (horizontal/vertical)
        """)
    
    with tab3:
        st.markdown("""
        ### 📊 Definición de Features
        
        **Métricas de Vibración:**
        
        - **RMS (Root Mean Square):** Métrica estándar ISO, representa energía total
        - **Vibración Máxima:** Amplitud pico, indica estrés máximo
        - **Vibración Media:** Nivel promedio, condición base
        - **Desviación Estándar:** Variabilidad, indica inestabilidad de señal
        - **Pico a Pico:** Excursión total, útil para evaluación de holgura
        
        **Diagnósticos Avanzados:**
        
        - **Kurtosis:** Mide impulsividad (normal ≈ 3, fallas en rodamientos > 6)
        - **Factor de Cresta:** Relación Pico/RMS (normal < 4, impactos > 6)
        - **Horas de Operación:** Factor de degradación basado en tiempo
        
        **Unidades:**
        - Velocidad: mm/s (milímetros por segundo)
        - Tiempo: horas (h)
        - Kurtosis y Factor de Cresta: adimensionales
        """)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Desarrollado por Michael Mancheno Medina</strong></p>
    <p>Ingeniero en Mantenimiento Industrial (EUR-ACE®) | Especialista en Machine Learning</p>
    <p>🔗 LinkedIn | 💻 GitHub | 📧 Email</p>
    <p style='font-size: 12px; margin-top: 10px;'>
        Basado en NASA IMS Bearing Dataset | Cumplimiento con ISO 10816-1 Grupo 2
    </p>
</div>
""", unsafe_allow_html=True)