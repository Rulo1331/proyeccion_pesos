"""
EL ROCIO - Proyección de Peso de Pollos
Dashboard comparativo de los modelos V3 y V4.

Archivos necesarios en la misma carpeta:
    modelos_v3.pkl   (generado por el notebook V3)
    modelos_v4.pkl   (generado por el notebook V4)
"""

import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------------------------------
# El bundle de V4 contiene objetos de la clase ModeloEnsamble definida en el notebook.
# joblib la busca en el módulo __main__, que bajo Streamlit NO es este script, así que
# hay que definirla aquí y registrarla a mano antes de cargar el archivo.
# ----------------------------------------------------------------------------------
class ModeloEnsamble:
    """Ensamble 50/50 XGBoost + Huber sobre el objetivo de ganancia relativa (V4)."""

    def predict(self, X):
        X = X[self.feats].astype(float)
        return (1 - self.w) * self.xgb.predict(X) + self.w * self.lin.predict(X.fillna(self.mediana))


sys.modules["__main__"].ModeloEnsamble = ModeloEnsamble

st.set_page_config(page_title="Predictor Avícola V3 vs V4", page_icon="🐔", layout="wide")

VERSIONES = {"V3": "modelos_v3.pkl", "V4": "modelos_v4.pkl"}
COLORES = {"V3": "#E08A1E", "V4": "#005088"}


# ==================================================================================
# CARGA
# ==================================================================================
@st.cache_resource
def cargar_bundles() -> dict:
    bundles = {}
    for nombre, ruta in VERSIONES.items():
        try:
            bundles[nombre] = joblib.load(ruta)
        except FileNotFoundError:
            st.error(f"No se encontró '{ruta}'. Súbelo a la misma carpeta que este script.")
    return bundles


bundles = cargar_bundles()
if len(bundles) == 0:
    st.stop()

REF = bundles.get("V4") or bundles.get("V3")
CONF = REF["config"]
DIAS = CONF["DIAS_PESAJE"]
DIAS_ACTUAL = CONF["DIAS_ACTUAL"]
DIAS_TARGET = CONF["DIAS_TARGET"]
ESTANDAR = REF["estandar"]
ALPHA = 1 - CONF["NIVEL_CONFIANZA"]


def std_dia(dia: int, sexo: str) -> float:
    return ESTANDAR[sexo][dia]


@st.cache_data
def catalogo(_bundles: dict) -> dict:
    """Granjas, galpones, corrales y procedencias conocidos por los modelos."""
    granjas, unidades, corrales, procs = set(), set(), set(), set()
    for b in _bundles.values():
        for mapas in b["te"].values():
            granjas.update(mapas["granja_te"][0].keys())
            unidades.update(mapas["galpon_te"][0].keys())
            procs.update(mapas["proc_te"][0].keys())
        for clave, tab in b["lags"].items():
            corrales.update(tab["corral"].keys())
    return {
        "granjas": sorted(granjas),
        "galpones": sorted({u.split("-", 1)[1] for u in unidades if "-" in u}),
        "corrales": sorted(corrales),          # tuplas (granja, galpon, corral)
        "procedencias": sorted(procs),
    }


CAT = catalogo(bundles)


# ==================================================================================
# MOTOR DE PREDICCIÓN (equivalente a proyectar() de los notebooks)
# ==================================================================================
def _te(valor, mapas, nombre):
    mapa, media_global = mapas[nombre]
    return mapa.get(valor, media_global)


def construir_fila(bundle, feats, pesos, sexo, granja, galpon, corral, proc, dia_actual, dia_target):
    """Arma la fila de variables que pide el modelo. Lo que no se puede conocer queda en NaN."""
    fila, faltantes = {}, []
    dias_con = [d for d in DIAS if d <= dia_actual and d in pesos]

    for d in dias_con:
        fila[f"ratio_d{d}"] = pesos[d] / std_dia(d, sexo)
    for a, b in zip(dias_con[:-1], dias_con[1:]):
        fila[f"gdp_rel_{a}_{b}"] = (pesos[b] - pesos[a]) / (std_dia(b, sexo) - std_dia(a, sexo))

    fila["sexo_enc"] = 1 if sexo == "M" else 0

    mapas = bundle["te"][(dia_actual, dia_target)]
    fila["granja_te"] = _te(granja, mapas, "granja_te")
    fila["galpon_te"] = _te(f"{granja}-{galpon}", mapas, "galpon_te")
    fila["proc_te"] = _te(proc, mapas, "proc_te")

    lags = bundle["lags"]
    clave_corral = (granja, galpon, corral)
    if dia_target in lags:
        fila[f"lag_corral_{dia_target}"] = lags[dia_target]["corral"].get(clave_corral, np.nan)
        fila[f"lag_granja_{dia_target}"] = lags[dia_target]["granja"].get(granja, np.nan)
    if (dia_actual, dia_target) in lags:                                    # solo V4
        tab = lags[(dia_actual, dia_target)]
        fila[f"lagd_corral_{dia_actual}_{dia_target}"] = tab["corral"].get(clave_corral, np.nan)
        fila[f"lagd_granja_{dia_actual}_{dia_target}"] = tab["granja"].get(granja, np.nan)

    X = pd.DataFrame([fila])
    for f in feats:
        if f not in X.columns:
            X[f] = np.nan
            faltantes.append(f)
    return X[feats], faltantes


def proyectar_lote(bundle, pesos, sexo, granja, galpon, corral, proc, dia_actual, dia_target):
    clave = (dia_actual, dia_target)
    modelo = bundle["modelos"][clave]
    feats = bundle["features"][clave]
    X, faltantes = construir_fila(bundle, feats, pesos, sexo, granja, galpon, corral, proc,
                                  dia_actual, dia_target)

    pred = float(modelo.predict(X)[0])
    if isinstance(modelo, ModeloEnsamble) or hasattr(modelo, "xgb"):
        pred += pesos[dia_actual] / std_dia(dia_actual, sexo)    # V4 predice la GANANCIA

    res = bundle["residuos"][clave]
    q = float(np.quantile(np.abs(res), 1 - ALPHA))
    std_t = std_dia(dia_target, sexo)
    return {
        "ratio": pred,
        "peso": pred * std_t,
        "std": std_t,
        "margen": (pred - 1) * std_t,
        "pct_std": (pred - 1) * 100,
        "lim_inf": (pred - q) * std_t,
        "lim_sup": (pred + q) * std_t,
        "prob": float(np.mean(pred + res >= 1.0)),
        "faltantes": faltantes,
    }


def estado(p: float) -> str:
    if p >= 0.75:
        return "✅ CUMPLE (probable)"
    if p >= 0.40:
        return "🟡 EN RIESGO"
    return "🔴 NO CUMPLE (probable)"


# ==================================================================================
# INTERFAZ
# ==================================================================================
def ui():
    st.title("🐔 EL ROCIO - Proyección de Peso · V3 vs V4")
    st.caption("Ingeniería de Procesos · comparación de ambos modelos sobre los mismos datos de entrada")

    if len(bundles) == 1:
        st.warning(f"Solo se cargó {list(bundles)[0]}. Para comparar, sube ambos archivos .pkl.")

    # ---------------- Sidebar ----------------
    st.sidebar.header("⚙️ Datos del lote")
    sexo = st.sidebar.selectbox("Sexo", ["M", "H"], help="M: macho, H: hembra")

    opciones_granja = CAT["granjas"] + ["➕ Otra / nueva"]
    sel = st.sidebar.selectbox("Granja", opciones_granja)
    granja = st.sidebar.text_input("Nombre de la granja nueva", "NUEVA") if sel == "➕ Otra / nueva" else sel

    galpones = sorted({g for (gr, g, _c) in CAT["corrales"] if gr == granja}) or CAT["galpones"]
    galpon = st.sidebar.selectbox("Galpón", galpones + ["➕ Otro"], index=0)
    if galpon == "➕ Otro":
        galpon = st.sidebar.text_input("Galpón nuevo", "01")
    galpon = str(galpon).zfill(2)

    corrales = sorted({c for (gr, g, c) in CAT["corrales"] if gr == granja and g == galpon})
    corral = st.sidebar.selectbox("Corral", corrales + ["➕ Otro"], index=0) if corrales else st.sidebar.text_input("Corral", "01")
    if corral == "➕ Otro":
        corral = st.sidebar.text_input("Corral nuevo", "01")
    corral = str(corral).zfill(2)

    proc = st.sidebar.selectbox("Granja reproductora (procedencia)",
                                ["(no indicar)"] + CAT["procedencias"])
    proc = None if proc == "(no indicar)" else proc

    st.sidebar.divider()
    st.sidebar.caption(
        "El historial del corral y de la granja se toma de los modelos entrenados. "
        "Si el corral no existe en el histórico, esas variables quedan vacías y el modelo lo resuelve igual."
    )

    # ---------------- Pesos ----------------
    st.subheader("📝 Pesos reales medidos (g)")
    st.caption("Deja en 0 los días que todavía no se han pesado.")
    cols = st.columns(len(DIAS_ACTUAL))
    pesos = {}
    for i, d in enumerate(DIAS_ACTUAL):
        with cols[i]:
            sugerido = round(std_dia(d, sexo)) + 5 if d <= 14 else 0
            val = st.number_input(f"Día {d}", min_value=0, value=sugerido, step=1, key=f"peso_{d}")
            if val > 0:
                pesos[d] = float(val)

    if not st.button("🚀 Generar proyección", type="primary"):
        return

    disponibles = [d for d in DIAS_ACTUAL if d in pesos]
    if not disponibles:
        st.error(f"Se necesita al menos un pesaje en los días {DIAS_ACTUAL}.")
        return
    dia_actual = max(disponibles)
    objetivos = [dt for dt in DIAS_TARGET if dt > dia_actual]
    if not objetivos:
        st.warning(f"Con pesaje hasta el día {dia_actual} no quedan días objetivo por proyectar.")
        return

    # ---------------- Cálculo ----------------
    res = {v: {} for v in bundles}
    faltantes = set()
    for v, b in bundles.items():
        for dt in objetivos:
            if (dia_actual, dt) not in b["modelos"]:
                continue
            try:
                r = proyectar_lote(b, pesos, sexo, granja, galpon, corral, proc, dia_actual, dt)
                res[v][dt] = r
                faltantes.update(r["faltantes"])
            except Exception as e:
                st.error(f"{v} · día {dt}: {e}")

    st.info(f"Proyectando desde el día **{dia_actual}** ({len(disponibles)} pesajes cargados).")

    # ---------------- Comparación principal ----------------
    st.subheader("📊 Comparación V3 vs V4")
    for dt in objetivos:
        r3, r4 = res.get("V3", {}).get(dt), res.get("V4", {}).get(dt)
        if not (r3 or r4):
            continue
        st.markdown(f"**Día {dt}**  ·  estándar {std_dia(dt, sexo):.0f} g")
        c1, c2, c3 = st.columns(3)
        if r3:
            c1.metric("V3", f"{r3['peso']:.0f} g", f"{r3['margen']:+.0f} g vs estándar")
            c1.caption(f"Intervalo 90 %: {r3['lim_inf']:.0f} – {r3['lim_sup']:.0f} g  ·  "
                       f"P(cumplir) {r3['prob']*100:.0f} %  {estado(r3['prob'])}")
        if r4:
            c2.metric("V4", f"{r4['peso']:.0f} g", f"{r4['margen']:+.0f} g vs estándar")
            c2.caption(f"Intervalo 90 %: {r4['lim_inf']:.0f} – {r4['lim_sup']:.0f} g  ·  "
                       f"P(cumplir) {r4['prob']*100:.0f} %  {estado(r4['prob'])}")
        if r3 and r4:
            dif = r4["peso"] - r3["peso"]
            c3.metric("Diferencia V4 − V3", f"{dif:+.0f} g", f"{dif / r3['peso'] * 100:+.1f} %",
                      delta_color="off")
            if abs(dif) > 0.05 * r3["peso"]:
                c3.warning("Discrepancia mayor al 5 %: conviene revisar los datos de entrada.")

    # ---------------- Gráfico ----------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DIAS, y=[std_dia(d, sexo) for d in DIAS],
                             name="Estándar", line=dict(dash="dash", color="grey")))
    dias_obs = sorted(pesos)
    fig.add_trace(go.Scatter(x=dias_obs, y=[pesos[d] for d in dias_obs], name="Pesajes reales",
                             mode="lines+markers", line=dict(color="#111111"), marker=dict(size=9)))
    for v in res:
        if not res[v]:
            continue
        dts = sorted(res[v])
        fig.add_trace(go.Scatter(
            x=[dia_actual] + dts,
            y=[pesos[dia_actual]] + [res[v][d]["peso"] for d in dts],
            name=f"Proyección {v}", mode="lines+markers",
            line=dict(color=COLORES[v], dash="dot"), marker=dict(size=11, symbol="diamond"),
            error_y=dict(type="data", symmetric=False,
                         array=[0] + [res[v][d]["lim_sup"] - res[v][d]["peso"] for d in dts],
                         arrayminus=[0] + [res[v][d]["peso"] - res[v][d]["lim_inf"] for d in dts],
                         thickness=1.2, width=6)))
    fig.update_layout(title="Curva de crecimiento y proyecciones",
                      xaxis_title="Día de edad", yaxis_title="Peso (g)", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    # ---------------- Tabla ----------------
    st.subheader("📋 Detalle")
    filas = []
    for d in sorted(pesos):
        filas.append({"Día": d, "Modelo": "Real", "Peso (g)": round(pesos[d], 1),
                      "Estándar": round(std_dia(d, sexo), 1),
                      "% vs Std": round((pesos[d] / std_dia(d, sexo) - 1) * 100, 1),
                      "Lím. inf.": None, "Lím. sup.": None, "P(cumplir)": None, "Estado": ""})
    for dt in objetivos:
        for v in ["V3", "V4"]:
            r = res.get(v, {}).get(dt)
            if not r:
                continue
            filas.append({"Día": dt, "Modelo": v, "Peso (g)": round(r["peso"], 1),
                          "Estándar": round(r["std"], 1), "% vs Std": round(r["pct_std"], 1),
                          "Lím. inf.": round(r["lim_inf"]), "Lím. sup.": round(r["lim_sup"]),
                          "P(cumplir)": f"{r['prob']*100:.0f} %", "Estado": estado(r["prob"])})
    st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)

    # ---------------- Transparencia ----------------
    with st.expander("ℹ️ Qué diferencia a V3 de V4 y qué variables no se pudieron completar"):
        st.markdown(
            "- **V3** predice directamente el peso relativo al estándar, con XGBoost.\n"
            "- **V4** predice la *ganancia* desde el último pesaje y la suma al peso actual. "
            "Usa un ensamble 50/50 de XGBoost y una regresión lineal robusta, "
            "más el historial de ganancia del corral y de la granja.\n"
            "- En el holdout (vueltas 204–207) el error promedio fue de 70.0 g en V3 y 65.3 g en V4."
        )
        if faltantes:
            st.markdown("**Variables que quedaron vacías en esta corrida** (el modelo las maneja como faltantes):")
            st.code("\n".join(sorted(faltantes)))
            if any(f.startswith("herm") for f in faltantes):
                st.caption(
                    "Las variables de 'lotes hermanos' solo existen cuando se procesa el Consolidado completo, "
                    "porque dependen de los demás lotes de la misma granja y vuelta. En esta interfaz, de ingreso "
                    "manual, no se pueden calcular."
                )


if __name__ == "__main__":
    ui()
