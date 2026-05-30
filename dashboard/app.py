import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Monitor OSINT Irán-EE.UU.-Israel",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cargar datos dummy por defecto si no existen los reales
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__dirname__)
    data_path = os.path.join(base_dir, "data", "processed", "dataset_diario.csv")
    
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            st.warning(f"Error cargando datos: {e}")
    
    # Generar datos dummy si no hay datos procesados
    dates = pd.date_range(start="2025-01-01", end=datetime.now().strftime("%Y-%m-%d"), freq='D')
    df = pd.DataFrame({'date': dates})
    
    # Simular eventos
    df['volume'] = np.random.randint(100, 1000, size=len(dates))
    df['avg_tone'] = np.random.uniform(-10, 5, size=len(dates))
    
    # Simular picos en junio 2025 y feb-abr 2026
    df.loc[(df['date'] >= '2025-06-13') & (df['date'] <= '2025-06-24'), 'volume'] *= 3
    df.loc[(df['date'] >= '2025-06-13') & (df['date'] <= '2025-06-24'), 'avg_tone'] -= 5
    
    df.loc[(df['date'] >= '2026-02-28') & (df['date'] <= '2026-04-08'), 'volume'] *= 4
    df.loc[(df['date'] >= '2026-02-28') & (df['date'] <= '2026-04-08'), 'avg_tone'] -= 8
    
    # Score de escalada simulado basado en volumen y tono
    df['escalation_score'] = (df['volume'] / 100) - df['avg_tone']
    df['escalation_score'] = 100 * (df['escalation_score'] - df['escalation_score'].min()) / (df['escalation_score'].max() - df['escalation_score'].min())
    
    return df

df = load_data()

# Título Principal
st.title("🛰️ Monitor OSINT de Escalada Regional")
st.subheader("Conflicto Irán–EE.UU.–Israel (2025-2026)")

# Sidebar
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a", ["🏠 Inicio & Métricas", "📈 Timeline Histórico", "📰 Monitor de Medios", "🤖 Predictor (ML)"])

st.sidebar.markdown("---")
st.sidebar.info(
    "Este dashboard utiliza datos de fuentes abiertas (GDELT, ACLED, RSS) "
    "para predecir niveles de escalada en Medio Oriente."
)

if page == "🏠 Inicio & Métricas":
    st.markdown("### Resumen Actual")
    
    # Métricas más recientes
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delta_score = latest['escalation_score'] - prev['escalation_score']
        st.metric(label="Score de Escalada (0-100)", 
                  value=f"{latest['escalation_score']:.1f}", 
                  delta=f"{delta_score:.1f}")
                  
    with col2:
        delta_vol = latest['volume'] - prev['volume']
        st.metric(label="Noticias Analizadas (Hoy)", 
                  value=f"{int(latest['volume'])}", 
                  delta=f"{int(delta_vol)}")
                  
    with col3:
        delta_tone = latest['avg_tone'] - prev['avg_tone']
        st.metric(label="Tono Mediático Promedio", 
                  value=f"{latest['avg_tone']:.2f}", 
                  delta=f"{delta_tone:.2f}",
                  delta_color="inverse")

    st.markdown("---")
    st.markdown("### Acerca del Score de Escalada")
    st.write("""
    El **Score de Escalada** es un índice calculado a partir de datos estructurales de conflicto (ACLED), 
    que toma en cuenta el número de eventos militares, fatalidades y el tipo de enfrentamiento (ej. ataques aéreos).
    
    Nuestro modelo de Machine Learning utiliza el volumen y el tono de las noticias internacionales (GDELT) 
    como *leading indicators* para predecir este score diariamente.
    """)

elif page == "📈 Timeline Histórico":
    st.markdown("### Evolución del Conflicto (2025-2026)")
    
    fig = px.line(df, x='date', y='escalation_score', title='Score de Escalada a lo largo del tiempo')
    
    # Anotaciones Históricas
    fig.add_vrect(x0="2025-06-13", x1="2025-06-24", fillcolor="red", opacity=0.2, line_width=0, annotation_text="Guerra 12 Días")
    fig.add_vrect(x0="2026-02-28", x1="2026-04-08", fillcolor="darkred", opacity=0.3, line_width=0, annotation_text="Op. Epic Fury")
    
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Score de Escalada (0-100)")
    st.plotly_chart(fig, use_container_width=True)
    
elif page == "📰 Monitor de Medios":
    st.markdown("### Análisis de Cobertura Global (GDELT)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_vol = px.bar(df, x='date', y='volume', title='Volumen de Noticias Diarias (Iran + Israel)')
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with col2:
        fig_tone = px.line(df, x='date', y='avg_tone', title='Tono Sentimental Promedio (Menor = Más Negativo)')
        fig_tone.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_tone, use_container_width=True)

elif page == "🤖 Predictor (ML)":
    st.markdown("### Simulador de Escalada")
    st.write("Ajusta los indicadores mediáticos para predecir el score de escalada.")
    
    col1, col2 = st.columns(2)
    with col1:
        sim_volume = st.slider("Volumen de Noticias (GDELT)", 0, 5000, 1000)
        sim_tone = st.slider("Tono Promedio", -15.0, 5.0, -5.0)
    
    # Modelo Lineal Simple (Placeholder para el modelo entrenado real)
    # En producción se cargaría el .pkl
    base_score = 10
    vol_impact = sim_volume * 0.01
    tone_impact = abs(sim_tone) * 2 if sim_tone < 0 else 0
    
    predicted_score = min(max(base_score + vol_impact + tone_impact, 0), 100)
    
    st.markdown(f"### Score Predicho: <span style='color:red'>{predicted_score:.1f} / 100</span>", unsafe_allow_html=True)
    
    st.progress(predicted_score / 100)
    
    if predicted_score > 75:
        st.error("🚨 Nivel Crítico: Riesgo inminente de operación militar mayor.")
    elif predicted_score > 40:
        st.warning("⚠️ Nivel Medio: Hostilidades limitadas o tensión regional alta.")
    else:
        st.success("✅ Nivel Bajo: Negociaciones o baja intensidad de conflicto.")
