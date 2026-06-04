import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predictor Avícola Pro", page_icon="🐔", layout="wide")

# --- LÓGICA DE NEGOCIO (Tu código adaptado) ---
STD_M = {0: 43, 4: 109, 7: 175, 10: 270, 14: 520, 21: 900, 28: 1550, 35: 2120}
STD_H = {0: 43, 4: 108, 7: 170, 10: 260, 14: 500, 21: 870, 28: 1480, 35: 2020}
MAE_MODELO = {21: 42, 28: 47, 35: 77}
DIAS_TODOS = [0, 4, 7, 10, 14, 21, 28, 35]
DIAS_TARGETS = [21, 28, 35]
TRAMOS = [(0,4),(4,7),(7,10),(10,14),(14,21),(21,28)]
DIAS_FEATURES = {21: [0, 4, 7, 10, 14], 28: [0, 4, 7, 10, 14, 21], 35: [0, 4, 7, 10, 14, 21, 28]}

def std_dia(dia, sexo):
    return STD_M[dia] if sexo == 'M' else STD_H[dia]

def construir_fila(pesos_completos, dia_target, sexo, galpon, granja):
    dias = DIAS_FEATURES[dia_target]
    row = {}
    for d in dias:
        std = std_dia(d, sexo)
        row[f'pesod{d}'] = pesos_completos[d]
        row[f'std_d{d}'] = std
        row[f'ratio_d{d}'] = pesos_completos[d] / std
        row[f'ok_d{d}'] = int(pesos_completos[d] >= std)
    for d_ini, d_fin in TRAMOS:
        if d_fin < dia_target and d_ini in pesos_completos and d_fin in pesos_completos:
            row[f'gdp_{d_ini}_{d_fin}'] = (pesos_completos[d_fin] - pesos_completos[d_ini]) / (d_fin - d_ini)
    row['sexo_enc'] = 1 if sexo == 'M' else 0
    row['galpon_enc'] = galpon - 1
    row['granja_enc'] = granja
    return pd.DataFrame([row])

# --- INTERFAZ STREAMLIT ---
st.title("🐔 Panel de Proyección de Peso en Cascada")
st.markdown("Estime el crecimiento de su lote mediante modelos de Machine Learning.")

# Sidebar - Parámetros fijos
st.sidebar.header("⚙️ Configuración del Lote")
sexo = st.sidebar.selectbox("Sexo del Lote", ["M", "H"], help="M: Macho, H: Hembra")
galpon = st.sidebar.number_input("Número de Galpón", min_value=1, value=1)
granja = st.sidebar.number_input("ID Granja (Encoding)", min_value=0, value=2)

# Sección de entrada de datos reales
st.subheader("📝 Pesos Reales Medidos")
cols = st.columns(5)
pesos_reales = {}

# Días fijos de entrada
dias_entrada = [0, 4, 7, 10, 14]
for i, d in enumerate(dias_entrada):
    with cols[i]:
        val = st.number_input(f"Día {d} (g)", min_value=0, value=std_dia(d, sexo) + 5)
        pesos_reales[d] = val

# Checkboxes para días opcionales (si ya los tienen)
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    if st.checkbox("Tengo peso de Día 21"):
        pesos_reales[21] = st.number_input("Día 21 Real (g)", value=950)
with col_opt2:
    if st.checkbox("Tengo peso de Día 28"):
        pesos_reales[28] = st.number_input("Día 28 Real (g)", value=1600)

if st.button("🚀 Generar Proyección", type="primary"):
    pesos = dict(pesos_reales)
    resultados = {}
    
    # Lógica de cascada
    ultimo_dia_real = max(pesos.keys())
    dias_a_proyectar = [d for d in DIAS_TARGETS if d > ultimo_dia_real]
    
    # Recopilar datos reales
    for d in DIAS_TODOS:
        if d in pesos:
            std = std_dia(d, sexo)
            resultados[d] = {
                'Peso': pesos[d], 'Fuente': 'Real', 
                'Margen': pesos[d] - std, 'Std': std
            }

    # Proyección
    with st.spinner('Ejecutando modelos en cascada...'):
        for dia_t in dias_a_proyectar:
            try:
                model = joblib.load(f'modelo_dia{dia_t}.pkl')
                features_names = joblib.load(f'features_dia{dia_t}.pkl')
                
                fila = construir_fila(pesos, dia_t, sexo, galpon, granja)
                pred = round(model.predict(fila[features_names])[0], 1)
                
                std = std_dia(dia_t, sexo)
                resultados[dia_t] = {
                    'Peso': pred, 'Fuente': 'Proyectado', 
                    'Margen': pred - std, 'Std': std
                }
                pesos[dia_t] = pred
            except Exception as e:
                st.error(f"Error en Día {dia_t}: {e}")
                import os
                st.write(f"Archivos presentes en el servidor: {os.listdir('.')}")

    # --- RESULTADOS VISUALES ---
    df_res = pd.DataFrame(resultados).T
    
    # KPI - Día 35
    if 35 in resultados:
        r35 = resultados[35]
        color = "normal" if r35['Margen'] >= 0 else "inverse"
        st.metric("Estimación Día 35", f"{r35['Peso']:.0f} g", f"{r35['Margen']:.1f} g vs Estándar", delta_color=color)

    # Gráfico
    fig = go.Figure()
    # Línea Estándar
    fig.add_trace(go.Scatter(x=DIAS_TODOS, y=[std_dia(d, sexo) for d in DIAS_TODOS], name="Estándar Mínimo", line=dict(dash='dash', color='grey')))
    # Línea Predicción
    fig.add_trace(go.Scatter(x=list(resultados.keys()), y=[r['Peso'] for r in resultados.values()], name="Trayectoria Lote", marker=dict(size=10, color='#005088')))
    
    fig.update_layout(title="Curva de Crecimiento del Lote", xaxis_title="Día", yaxis_title="Peso (g)")
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de Datos
    st.table(
    df_res.style
    .format({
        'Peso': '{:.1f}', 
        'Margen': '{:.1f}', 
        'Std': '{:.1f}'
    })
    .map(lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: black', subset=['Margen'])
)
