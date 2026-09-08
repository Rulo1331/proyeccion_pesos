import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predictor Avícola V2.1", page_icon="🐔", layout="wide")

RUTA_BUNDLE = "modelos_v2.pkl"
COL_GRANJA = "Granja"
COL_GALPON = "Galpon"
COL_SEXO = "Sexo"


# --- CARGA DEL BUNDLE (todo sale de aquí: días, estándar, config, TE) ---
@st.cache_resource
def cargar_bundle(ruta: str):
    return joblib.load(ruta)


try:
    bundle = cargar_bundle(RUTA_BUNDLE)
except FileNotFoundError:
    st.error(f"No se encontró '{RUTA_BUNDLE}'. Colócalo en la misma carpeta que este script.")
    st.stop()

DIAS = bundle["dias"]                       # p.ej. [0,4,7,10,14,21,28,35]
ESTANDAR = bundle["estandar"]                # {'M': {dia: std}, 'H': {dia: std}}
CONF = bundle["config"]
DIAS_ACTUAL = CONF["DIAS_ACTUAL"]            # días válidos como "último pesaje" (no incluye 0)
DIAS_TARGET = CONF["DIAS_TARGET"]            # p.ej. [21, 28, 35]
ALPHA = 1 - CONF["NIVEL_CONFIANZA"]


def std_dia(dia: int, sexo: str) -> float:
    return ESTANDAR[sexo][dia]


def aplicar_te(valor, mapa: dict, media_global: float) -> float:
    # Granja/galpón no visto en entrenamiento -> media global (igual que en el notebook)
    return mapa.get(valor, media_global)


def _estado(p: float) -> str:
    if p >= 0.75:
        return "✅ CUMPLE (probable)"
    if p >= 0.40:
        return "🟡 EN RIESGO"
    return "🔴 NO CUMPLE (probable)"


@st.cache_data
def granjas_conocidas(_bundle) -> list:
    nombres = set()
    for (_da, _dt), (mapa_g, _glob) in _bundle["te_granja"].items():
        nombres.update(mapa_g.keys())
    return sorted(nombres)


def proyectar_lote(pesos: dict, sexo: str, granja: str, galpon: str, dia_actual: int, dia_target: int) -> dict:
    """Replica proyectar() del notebook para un solo lote y un solo horizonte."""
    modelo = bundle["modelos"][(dia_actual, dia_target)]
    feats = bundle["features"][(dia_actual, dia_target)]
    mapa_g, glob_g = bundle["te_granja"][(dia_actual, dia_target)]
    mapa_u, glob_u = bundle["te_galpon"][(dia_actual, dia_target)]

    dias_con = [d for d in DIAS if d <= dia_actual and d in pesos]

    row = {}
    for d in dias_con:
        row[f"ratio_d{d}"] = pesos[d] / std_dia(d, sexo)
    for a, b in zip(dias_con[:-1], dias_con[1:]):
        gan_obs = pesos[b] - pesos[a]
        gan_std = std_dia(b, sexo) - std_dia(a, sexo)
        row[f"gdp_rel_{a}_{b}"] = gan_obs / gan_std

    row["sexo_enc"] = 1 if sexo == "M" else 0
    unidad = f"{granja}-{galpon}"
    row["granja_te"] = aplicar_te(granja, mapa_g, glob_g)
    row["galpon_te"] = aplicar_te(unidad, mapa_u, glob_u)

    X = pd.DataFrame([row])
    # columnas faltantes (día intermedio sin pesaje) -> NaN, XGBoost las maneja nativamente
    for f in feats:
        if f not in X.columns:
            X[f] = np.nan
    X = X[feats]

    pred_ratio = float(modelo.predict(X)[0])
    res = bundle["residuos"][(dia_actual, dia_target)]
    q = np.quantile(np.abs(res), 1 - ALPHA)
    prob_cumplir = float(np.mean(pred_ratio + res >= 1.0))
    std_t = std_dia(dia_target, sexo)

    return {
        "Peso": round(pred_ratio * std_t, 1),
        "Fuente": "Proyectado",
        "Std": round(std_t, 1),
        "Margen": round((pred_ratio - 1) * std_t, 1),
        "% Std": round(pred_ratio * 100, 1),
        "lim_inf": round((pred_ratio - q) * std_t, 1),
        "lim_sup": round((pred_ratio + q) * std_t, 1),
        "prob_cumplir": round(prob_cumplir, 2),
        "estado": _estado(prob_cumplir),
    }


# --- INTERFAZ STREAMLIT ---
st.title("🐔 EL ROCIO - Proyección de Peso (V2.1)")
st.markdown("Ingeniería de Procesos")

# Sidebar - Parámetros del lote
st.sidebar.header("⚙️ Configuración del Lote")
sexo = st.sidebar.selectbox("Sexo del Lote", ["M", "H"], help="M: Macho, H: Hembra")
galpon = st.sidebar.text_input("Galpón", value="1")

opciones_granja = granjas_conocidas(bundle) + ["➕ Otra / granja nueva"]
sel_granja = st.sidebar.selectbox("Granja", options=opciones_granja)
if sel_granja == "➕ Otra / granja nueva":
    granja = st.sidebar.text_input("Nombre de la granja nueva", value="NUEVA")
    st.sidebar.caption("Granja no vista en entrenamiento: se usará el promedio histórico global (target encoding).")
else:
    granja = sel_granja

# Sección de entrada de datos reales
st.subheader("📝 Pesos Reales Medidos")
dias_entrada = [d for d in DIAS if d in (0, 4, 7, 10, 14)]
cols = st.columns(len(dias_entrada))
pesos_reales = {}
for i, d in enumerate(dias_entrada):
    with cols[i]:
        val = st.number_input(f"Día {d} (g)", min_value=0, value=round(std_dia(d, sexo)) + 5, key=f"peso_{d}")
        pesos_reales[d] = val

# Checkboxes para días opcionales ya pesados
col_opt1, col_opt2 = st.columns(2)
if 21 in DIAS_ACTUAL:
    with col_opt1:
        if st.checkbox("Tengo peso de Día 21"):
            pesos_reales[21] = st.number_input("Día 21 Real (g)", value=round(std_dia(21, sexo)))
if 28 in DIAS_ACTUAL:
    with col_opt2:
        if st.checkbox("Tengo peso de Día 28"):
            pesos_reales[28] = st.number_input("Día 28 Real (g)", value=round(std_dia(28, sexo)))

if st.button("🚀 Generar Proyección", type="primary"):
    pesos = dict(pesos_reales)

    # Último pesaje válido como "día actual" para el modelo (debe estar en DIAS_ACTUAL)
    dias_actuales_disponibles = [d for d in DIAS_ACTUAL if d in pesos]
    if not dias_actuales_disponibles:
        st.error(f"Se necesita al menos un pesaje en los días {DIAS_ACTUAL} para poder proyectar.")
        st.stop()
    dia_actual = max(dias_actuales_disponibles)

    resultados = {}
    for d, p in pesos.items():
        std = std_dia(d, sexo)
        resultados[d] = {"Peso": p, "Fuente": "Real", "Margen": round(p - std, 1), "Std": round(std, 1),
                          "% Std": round(p / std * 100, 1),
                          "lim_inf": None, "lim_sup": None, "prob_cumplir": None, "estado": ""}

    dias_a_proyectar = [dt for dt in DIAS_TARGET if dt > dia_actual and (dia_actual, dt) in bundle["modelos"]]
    if not dias_a_proyectar:
        st.warning(f"No hay modelo disponible para proyectar desde el día {dia_actual}.")

    with st.spinner("Ejecutando proyección..."):
        for dt in dias_a_proyectar:
            try:
                resultados[dt] = proyectar_lote(pesos, sexo, granja, galpon, dia_actual, dt)
            except Exception as e:
                st.error(f"Error proyectando día {dt}: {e}")

    # --- RESULTADOS VISUALES ---
    df_res = pd.DataFrame(resultados).T.sort_index()

    # KPI - último día objetivo proyectado
    if dias_a_proyectar:
        dt_final = max(dias_a_proyectar)
        r_final = resultados[dt_final]
        color = "normal" if r_final["Margen"] >= 0 else "inverse"
        st.metric(
            f"Estimación Día {dt_final}",
            f"{r_final['Peso']:.0f} g",
            f"{r_final['Margen']:.1f} g vs Estándar",
            delta_color=color,
        )
        c1, c2 = st.columns(2)
        c1.metric("Intervalo de predicción (90%)", f"{r_final['lim_inf']:.0f} – {r_final['lim_sup']:.0f} g")
        c2.metric("Probabilidad de cumplir estándar", f"{r_final['prob_cumplir']*100:.0f}%", r_final["estado"])

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=DIAS, y=[std_dia(d, sexo) for d in DIAS],
        name="Estándar Mínimo", line=dict(dash="dash", color="grey"),
    ))
    dias_ordenados = sorted(resultados.keys())
    fig.add_trace(go.Scatter(
        x=dias_ordenados, y=[resultados[d]["Peso"] for d in dias_ordenados],
        name="Trayectoria Lote", mode="lines+markers", marker=dict(size=10, color="#005088"),
    ))
    # Barras de error para los días proyectados (intervalo de predicción)
    dias_proy = [d for d in dias_ordenados if resultados[d]["Fuente"] == "Proyectado"]
    if dias_proy:
        fig.add_trace(go.Scatter(
            x=dias_proy,
            y=[resultados[d]["Peso"] for d in dias_proy],
            error_y=dict(
                type="data", symmetric=False,
                array=[resultados[d]["lim_sup"] - resultados[d]["Peso"] for d in dias_proy],
                arrayminus=[resultados[d]["Peso"] - resultados[d]["lim_inf"] for d in dias_proy],
            ),
            mode="markers", marker=dict(size=1, color="#C0392B"),
            name="Intervalo 90%", showlegend=True,
        ))
    fig.update_layout(title="Curva de Crecimiento del Lote", xaxis_title="Día", yaxis_title="Peso (g)")
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de Datos
    cols_tabla = ["Fuente", "Peso", "Std", "% Std", "Margen", "lim_inf", "lim_sup", "prob_cumplir", "estado"]
    st.table(
        df_res[cols_tabla]
        .rename(columns={"lim_inf": "Lím. Inf.", "lim_sup": "Lím. Sup.",
                          "prob_cumplir": "Prob. Cumplir", "estado": "Estado"})
        .style
        .format({"Peso": "{:.1f}", "Margen": "{:.1f}", "Std": "{:.1f}",
                  "% Std": lambda x: f"{x:.1f}%" if pd.notna(x) else "",
                  "Lím. Inf.": lambda x: f"{x:.0f}" if pd.notna(x) else "",
                  "Lím. Sup.": lambda x: f"{x:.0f}" if pd.notna(x) else "",
                  "Prob. Cumplir": lambda x: f"{x*100:.0f}%" if pd.notna(x) else ""})
        .map(lambda x: "color: red" if isinstance(x, (int, float)) and x < 0 else "color: black", subset=["Margen"])
        .map(lambda x: "color: red" if isinstance(x, (int, float)) and x < 100 else "color: green", subset=["% Std"])
    )
