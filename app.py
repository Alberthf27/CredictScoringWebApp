"""
Credit Scoring - German Credit Data
App Streamlit con diseno profesional, selector inteligente de columnas,
y mapeo flexible de nombres (espanol/ingles/variantes).
"""

import io
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import arff


# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="Credit Scoring | German Credit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS - PROFESIONAL Y LIMPIO
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ============================================
       BASE
       ============================================ */
    html, body, [class*="css"], .stMarkdown, p, span, div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; height: 0; }

    /* Layout principal centrado y amplio */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1200px;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    [data-testid="stAppViewContainer"] > .main {
        max-width: 1200px;
        margin: 0 auto;
    }

    /* ============================================
       TIPOGRAFÍA (más grande y legible)
       ============================================ */
    h1 { font-size: 2.4rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.6rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.2rem !important; font-weight: 700 !important; }
    p, li { font-size: 0.95rem !important; line-height: 1.6 !important; }
    label, .stMarkdown { font-size: 0.95rem !important; }

    /* ============================================
       HEADER
       ============================================ */
    .app-header {
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid #e5e9f0;
        margin-bottom: 2rem;
    }
    .app-header h1 {
        color: #1a365d;
        margin: 0;
        letter-spacing: -0.025em;
    }
    .app-header p {
        color: #718096;
        margin: 0.4rem 0 0 0;
        font-size: 1.05rem;
    }

    /* ============================================
       SECTION TITLES
       ============================================ */
    .section-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.8rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #edf2f7;
    }

    /* ============================================
       CARDS
       ============================================ */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1a365d;
        margin: 0 0 0.6rem 0;
    }
    .card-desc {
        font-size: 0.88rem;
        color: #718096;
        line-height: 1.5;
        margin: 0;
    }

    /* ============================================
       STAT CARDS (más grandes)
       ============================================ */
    .stat-card {
        background: #ffffff;
        padding: 1.3rem 1.4rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    .stat-card .label {
        color: #718096;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .stat-card .value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.3rem 0 0 0;
        color: #1a365d;
    }
    .stat-card .delta {
        color: #718096;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    .stat-card.success { border-left: 4px solid #16a34a; }
    .stat-card.success .value { color: #15803d; }
    .stat-card.danger { border-left: 4px solid #dc2626; }
    .stat-card.danger .value { color: #b91c1c; }
    .stat-card.warning { border-left: 4px solid #f59e0b; }
    .stat-card.warning .value { color: #b45309; }
    .stat-card.info { border-left: 4px solid #3b82f6; }
    .stat-card.info .value { color: #1e40af; }

    /* ============================================
       ALERTAS
       ============================================ */
    .info-card {
        background: #ebf8ff;
        border-left: 4px solid #3182ce;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        color: #2c5282;
        font-size: 0.9rem;
        margin: 0.6rem 0;
    }
    .success-card {
        background: #f0fff4;
        border-left: 4px solid #38a169;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        color: #22543d;
        font-size: 0.9rem;
        margin: 0.6rem 0;
    }
    .error-card {
        background: #fff5f5;
        border-left: 4px solid #e53e3e;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        color: #742a2a;
        font-size: 0.9rem;
        margin: 0.6rem 0;
    }
    .warning-card {
        background: #fffaf0;
        border-left: 4px solid #ed8936;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        color: #7b341e;
        font-size: 0.9rem;
        margin: 0.6rem 0;
    }

    /* ============================================
       FILE UPLOADER
       ============================================ */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px dashed #cbd5e0;
        border-radius: 10px;
        padding: 2rem;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        border-color: #3182ce;
    }
    [data-testid="stFileUploaderDropzone"] section {
        color: #2d3748;
    }

    /* ============================================
       RADIO BUTTONS
       ============================================ */
    .stRadio > label {
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
    .stRadio [role="radiogroup"] {
        gap: 0.5rem;
    }

    /* ============================================
       BOTONES
       ============================================ */
    .stButton>button {
        background: #1a365d !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-radius: 8px !important;
        padding: 0.7rem 1.8rem !important;
        border: none !important;
        box-shadow: 0 1px 3px rgba(26, 54, 93, 0.2) !important;
    }
    .stButton>button:hover {
        background: #2c5282 !important;
        box-shadow: 0 4px 12px rgba(26, 54, 93, 0.3) !important;
    }
    .stDownloadButton>button {
        background: #16a34a !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-radius: 8px !important;
        padding: 0.7rem 1.8rem !important;
        border: none !important;
        box-shadow: 0 1px 3px rgba(22, 163, 74, 0.2) !important;
    }
    .stDownloadButton>button:hover {
        background: #15803d !important;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.3) !important;
    }

    /* ============================================
       TABS
       ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        padding: 0.7rem 1.3rem;
        font-weight: 600;
        color: #718096;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #1a365d !important;
        border-bottom: 3px solid #1a365d;
        margin-bottom: -2px;
    }

    /* ============================================
       MULTISELECT
       ============================================ */
    .stMultiSelect [data-baseweb="tag"] {
        background: #ebf8ff !important;
        color: #2c5282 !important;
        border: none !important;
    }

    /* ============================================
       METRIC
       ============================================ */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: #718096 !important;
    }

    /* Code */
    code {
        background: #edf2f7;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-size: 0.88em;
        color: #2c5282;
        font-weight: 500;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* Espaciado general entre elementos */
    .element-container { margin-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CARGA DE MODELOS
# ============================================================
@st.cache_resource(show_spinner="Cargando modelo...")
def load_models():
    try:
        return (
            joblib.load("preprocessor.pkl"),
            joblib.load("random_forest_model.pkl"),
            joblib.load("feature_names.pkl"),
            joblib.load("categorical_options.pkl"),
            joblib.load("feature_defaults.pkl"),
        )
    except FileNotFoundError as e:
        st.error(
            f"❌ Archivo no encontrado: **{e.filename}**\n\n"
            "Asegúrate de que los 5 archivos `.pkl` estén junto a `app.py`:\n"
            "- `preprocessor.pkl`\n"
            "- `random_forest_model.pkl`\n"
            "- `feature_names.pkl`\n"
            "- `categorical_options.pkl`\n"
            "- `feature_defaults.pkl`"
        )
        st.stop()


preprocessor, model, feature_names, categorical_options, feature_defaults = load_models()


# ============================================================
# MAPEO DE NOMBRES DE COLUMNAS (FLEXIBLE)
# ============================================================
# Mapeo: nombre en el archivo del usuario -> nombre interno del modelo
MAPEO_NOMBRES = {
    # Age
    "age": "age", "edad": "age", "años": "age", "anos": "age", "year": "age", "years": "age",
    # Sex / personal_status
    "sex": "personal_status", "sexo": "personal_status", "genero": "personal_status",
    "gender": "personal_status", "personal_status": "personal_status",
    "personal status": "personal_status", "estado_civil": "personal_status",
    # Job
    "job": "job", "trabajo": "job", "empleo": "job", "ocupacion": "job", "occupation": "job",
    # Housing
    "housing": "housing", "vivienda": "housing", "casa": "housing", "house": "housing",
    # Saving accounts
    "saving accounts": "savings_status", "savings": "savings_status",
    "saving_accounts": "savings_status", "savings_status": "savings_status",
    "ahorros": "savings_status", "cuenta de ahorros": "savings_status",
    "cuenta_de_ahorros": "savings_status", "ahorro": "savings_status",
    # Checking account
    "checking account": "checking_status", "checking": "checking_status",
    "checking_account": "checking_status", "checking_status": "checking_status",
    "cuenta corriente": "checking_status", "corriente": "checking_status",
    "cuenta_corriente": "checking_status",
    # Credit amount
    "credit amount": "credit_amount", "credit_amount": "credit_amount", "amount": "credit_amount",
    "monto": "credit_amount", "monto del credito": "credit_amount", "monto_del_credito": "credit_amount",
    "monto de credito": "credit_amount", "credito": "credit_amount", "credit": "credit_amount",
    "credit_amount": "credit_amount",
    # Duration
    "duration": "duration", "duracion": "duration", "plazo": "duration", "meses": "duration", "months": "duration",
    # Purpose
    "purpose": "purpose", "proposito": "purpose", "motivo": "purpose", "objetivo": "purpose",
    "reason": "purpose",
}

# Las 9 columnas del modo simplificado
COLUMNAS_SIMPLIFICADAS = [
    "age", "personal_status", "job", "housing",
    "savings_status", "checking_status",
    "credit_amount", "duration", "purpose",
]


def normalizar_columna(col: str) -> str:
    """Normaliza un nombre de columna: lowercase, sin acentos, sin espacios extra."""
    s = str(col).strip().lower()
    # Quitar acentos
    s = (s.replace("á", "a").replace("é", "e").replace("í", "i")
           .replace("ó", "o").replace("ú", "u").replace("ñ", "n"))
    # Quitar guiones bajos y reemplazar por espacios
    s = re.sub(r"[_\-]+", " ", s)
    # Quitar espacios múltiples
    s = re.sub(r"\s+", " ", s)
    return s


def mapear_columnas(df: pd.DataFrame) -> tuple:
    """Mapea columnas del usuario a nombres internos. Devuelve (df_mapeado, mapeo_info)."""
    df = df.copy()
    rename = {}
    info = []
    for col_original in df.columns:
        col_norm = normalizar_columna(col_original)
        if col_norm in MAPEO_NOMBRES:
            target = MAPEO_NOMBRES[col_norm]
            if target not in rename.values():
                rename[col_original] = target
                info.append({"original": col_original, "interno": target, "match": "exacto"})
            else:
                info.append({"original": col_original, "interno": None, "match": "duplicado"})
        else:
            # Intentar match parcial
            for key, val in MAPEO_NOMBRES.items():
                if key in col_norm or col_norm in key:
                    if val not in rename.values():
                        rename[col_original] = val
                        info.append({"original": col_original, "interno": val, "match": "parcial"})
                        break
            else:
                info.append({"original": col_original, "interno": None, "match": "sin match"})
    df = df.rename(columns=rename)
    return df, info


# ============================================================
# UTILIDADES DE CARGA DE ARCHIVOS
# ============================================================
COLUMNAS_UCI = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment", "installment_commitment",
    "personal_status", "other_parties", "residence_since", "property_magnitude",
    "age", "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]

MAPEO_CODIGOS_UCI = {
    "checking_status": {"A11": "<0", "A12": "0<=X<200", "A13": ">=200", "A14": "no checking"},
    "credit_history": {
        "A30": "no credits/all paid", "A31": "all paid", "A32": "existing paid",
        "A33": "delayed previously", "A34": "critical/other existing credit",
    },
    "purpose": {
        "A40": "new car", "A41": "used car", "A42": "furniture/equipment",
        "A43": "radio/tv", "A44": "domestic appliance", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining",
        "A49": "business", "A410": "other",
    },
    "savings_status": {
        "A61": "<100", "A62": "100<=X<500", "A63": "500<=X<1000",
        "A64": ">=1000", "A65": "no known savings",
    },
    "employment": {
        "A71": "unemployed", "A72": "<1", "A73": "1<=X<4",
        "A74": "4<=X<7", "A75": ">=7",
    },
    "personal_status": {
        "A91": "male div/sep", "A92": "female div/dep/mar",
        "A93": "male single", "A94": "male mar/wid", "A95": "female single",
    },
    "other_parties": {"A101": "none", "A102": "co applicant", "A103": "guarantor"},
    "property_magnitude": {
        "A121": "real estate", "A122": "life insurance",
        "A123": "car", "A124": "no known property",
    },
    "other_payment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemp/unskilled non res", "A172": "unskilled resident",
        "A173": "skilled", "A174": "high qualif/self emp/mgmt",
    },
    "own_telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


def decodificar_codigos_uci(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapeo in MAPEO_CODIGOS_UCI.items():
        if col in df.columns:
            valores_unicos = set(df[col].astype(str).unique())
            if valores_unicos & set(mapeo.keys()):
                df[col] = df[col].astype(str).map(lambda x: mapeo.get(x, x))
    if "class" in df.columns:
        vals = set(df["class"].astype(str).unique())
        if vals & {"1", "2"}:
            df["class"] = df["class"].astype(str).map({"1": "good", "2": "bad"}).fillna(df["class"])
    return df


def decodificar_inteligente(contenido_bytes: bytes) -> str:
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]:
        try:
            return contenido_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return contenido_bytes.decode("latin-1", errors="replace")


def _probar_sin_header(texto: str, sep: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(io.StringIO(texto), sep=sep, skipinitialspace=True,
                         skip_blank_lines=True, comment="#", header=None, on_bad_lines="skip")
        if df.shape[1] in (20, 21):
            if df.shape[1] == 21:
                df.columns = COLUMNAS_UCI
            else:
                df.columns = COLUMNAS_UCI[:-1]
            return df
    except Exception:
        return None
    return None


def _probar_con_header(texto: str, sep: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(io.StringIO(texto), sep=sep, skipinitialspace=True,
                         skip_blank_lines=True, comment="#", header=0, on_bad_lines="skip")
        if 20 <= len(df.columns) <= 25:
            sample_cols = [str(c) for c in df.columns[:5]]
            if any(v.startswith("A") and v[1:].isdigit() for v in sample_cols):
                return None
            return df
    except Exception:
        return None
    return None


def cargar_archivo_robusto(archivo) -> pd.DataFrame:
    nombre = archivo.name.lower()
    if nombre.endswith(".arff"):
        contenido = archivo.read().decode("utf-8")
        datos, _ = arff.loadarff(io.StringIO(contenido))
        df = pd.DataFrame(datos)
        for col in df.select_dtypes([object]):
            df[col] = df[col].str.decode("utf-8")
        return decodificar_codigos_uci(df)

    contenido_bytes = archivo.read()
    archivo.seek(0)
    texto = decodificar_inteligente(contenido_bytes)

    for sep in [" ", "\t", "|", ":"]:
        df = _probar_sin_header(texto, sep)
        if df is not None:
            return decodificar_codigos_uci(df)

    for sep in [",", ";", "\t", " ", "|", ":"]:
        df = _probar_con_header(texto, sep)
        if df is not None:
            return decodificar_codigos_uci(df)

    for sep in [",", ";", " ", "|", ":"]:
        df = _probar_sin_header(texto, sep)
        if df is not None:
            return decodificar_codigos_uci(df)

    raise ValueError("No se pudo parsear el archivo. Verifica el formato.")


def aplicar_defaults(df: pd.DataFrame) -> tuple:
    """Llena features faltantes con defaults. Devuelve (df_completo, defaults_usados)."""
    df = df.copy()
    defaults_usados = []
    for col in feature_names:
        if col not in df.columns:
            df[col] = feature_defaults[col]
            defaults_usados.append(col)
    return df, defaults_usados


# ============================================================
# PREDICCIÓN
# ============================================================
def predecir_lote(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_names].copy()
    X_t = preprocessor.transform(X)
    preds = model.predict(X_t)
    probas = model.predict_proba(X_t)

    clases = model.classes_
    idx_good = list(clases).index("good")
    idx_bad = list(clases).index("bad")

    resultados = df.copy()
    resultados["Predicción"] = ["✅ Solvente" if p == "good" else "⚠️ Moroso" for p in preds]
    resultados["Prob. Solvente (%)"] = (probas[:, idx_good] * 100).round(2)
    resultados["Prob. Moroso (%)"] = (probas[:, idx_bad] * 100).round(2)
    resultados["Nivel de Riesgo"] = pd.cut(
        resultados["Prob. Moroso (%)"],
        bins=[0, 30, 50, 70, 100],
        labels=["Bajo", "Moderado", "Alto", "Muy Alto"],
    )
    return resultados


def colorear_prediccion(val):
    if "Solvente" in str(val):
        return "background-color: #c6f6d5; color: #22543d; font-weight: 600"
    if "Moroso" in str(val):
        return "background-color: #fed7d7; color: #742a2a; font-weight: 600"
    return ""


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="app-header">
        <h1>🏦 Credit Scoring</h1>
        <p>Predicción de riesgo crediticio · German Credit Data · Random Forest</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SELECTOR DE MODO
# ============================================================
st.markdown('<p class="section-title">1. Modo de análisis</p>', unsafe_allow_html=True)

modo = st.radio(
    "Selecciona el modo",
    options=["Completo (20 columnas)", "Simplificado (9 columnas)", "Personalizado"],
    horizontal=True,
    label_visibility="collapsed",
)

if modo == "Completo (20 columnas)":
    st.markdown(
        '<div class="info-card">El archivo debe contener las <b>20 columnas</b> del modelo. '
        "Si tu archivo tiene menos, usa el modo Simplificado o Personalizado.</div>",
        unsafe_allow_html=True,
    )
    columnas_requeridas = feature_names
elif modo == "Simplificado (9 columnas)":
    st.markdown(
        '<div class="info-card">Tu archivo necesita solo estas <b>9 columnas</b> '
        "(reconocemos variaciones como <code>Edad</code>/<code>Age</code>, "
        "<code>Sexo</code>/<code>Sex</code>, etc.). Las 11 restantes se completan automáticamente.<br><br>"
        "<b>Columnas reconocidas:</b> "
        "<code>Age</code>, <code>Sex</code>, <code>Job</code>, <code>Housing</code>, "
        "<code>Saving accounts</code>, <code>Checking account</code>, "
        "<code>Credit amount</code>, <code>Duration</code>, <code>Purpose</code>"
        "</div>",
        unsafe_allow_html=True,
    )
    columnas_requeridas = COLUMNAS_SIMPLIFICADAS
else:  # Personalizado
    cols_sel = st.multiselect(
        "Selecciona las columnas que tendrá tu archivo",
        options=feature_names,
        default=feature_names,
        help="Las columnas no seleccionadas se completan con valores por defecto del training set",
    )
    columnas_requeridas = cols_sel
    if len(cols_sel) == 0:
        st.markdown('<div class="warning-card">⚠️ Selecciona al menos una columna.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="info-card">Tu archivo tendrá <b>{len(cols_sel)} columnas</b>. '
            f"Las {20 - len(cols_sel)} restantes se completan automáticamente con valores típicos del dataset.</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# UPLOAD
# ============================================================
st.markdown('<p class="section-title">2. Cargar archivo</p>', unsafe_allow_html=True)

archivo_subido = st.file_uploader(
    "Arrastra o selecciona tu archivo CSV/ARFF",
    type=["csv", "arff", "txt", "data"],
    help="Formatos soportados: CSV (con o sin headers, formato UCI), ARFF.",
    label_visibility="collapsed",
)

# Procesar
if archivo_subido is not None and len(columnas_requeridas) > 0:
    try:
        with st.spinner("Leyendo archivo..."):
            df_input = cargar_archivo_robusto(archivo_subido)
    except Exception as e:
        st.markdown(
            f'<div class="error-card"><b>No se pudo leer el archivo</b><br><br>{e}<br><br>'
            "Verifica que sea un CSV o ARFF válido.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    # Quitar columna target si existe
    if "class" in df_input.columns:
        df_input = df_input.drop(columns=["class"])

    # Mapear nombres de columnas (amigables -> internos)
    df_input, info_mapeo = mapear_columnas(df_input)

    # Verificar columnas presentes
    presentes = [c for c in columnas_requeridas if c in df_input.columns]
    faltantes = [c for c in columnas_requeridas if c not in df_input.columns]

    if not presentes:
        st.markdown(
            f'<div class="error-card"><b>Ninguna columna reconocida</b><br><br>'
            f"Tu archivo tiene: {', '.join(f'<code>{c}</code>' for c in df_input.columns[:10])}<br><br>"
            f"<b>Esperadas ({len(columnas_requeridas)}):</b> "
            + ", ".join(f"<code>{c}</code>" for c in columnas_requeridas)
            + "</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    # Mostrar tabla de mapeo
    with st.expander(f"🔍 Mapeo de columnas ({len(presentes)} reconocidas, {len(faltantes)} por defecto)", expanded=False):
        mapeo_df = pd.DataFrame([
            {"Tu columna": m["original"], "Mapeada a": m["interno"] or "—", "Tipo": m["match"]}
            for m in info_mapeo
        ])
        st.dataframe(mapeo_df, use_container_width=True, hide_index=True)

    # Si modo completo, requerir las 20
    if modo == "Completo (20 columnas)" and len(presentes) < 20:
        st.markdown(
            f'<div class="error-card"><b>Modo completo requiere las 20 columnas</b><br><br>'
            f"Solo se encontraron {len(presentes)} de 20. Faltan: "
            + ", ".join(f"<code>{c}</code>" for c in faltantes[:5])
            + ("..." if len(faltantes) > 5 else "")
            + '<br><br>💡 Cambia a modo <b>Simplificado</b> o <b>Personalizado</b>.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # Aplicar defaults
    df_completo, defaults_usados = aplicar_defaults(df_input)

    if defaults_usados:
        st.markdown(
            f'<div class="info-card"><b>{len(defaults_usados)} columnas</b> completadas con valores por defecto: '
            + ", ".join(f"<code>{c}</code>" for c in defaults_usados[:5])
            + ("..." if len(defaults_usados) > 5 else "")
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="success-card"><b>Listo</b> · {len(df_input)} clientes · '
        f"{len(presentes)} columnas del archivo + {len(defaults_usados)} por defecto</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Vista previa (primeras 5 filas)", expanded=False):
        st.dataframe(df_input.head(5), use_container_width=True, hide_index=True)

    # Botón de predicción
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Predecir", type="primary", use_container_width=True):
            with st.spinner("Procesando..."):
                try:
                    st.session_state["resultados"] = predecir_lote(df_completo)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()


# ============================================================
# RESULTADOS
# ============================================================
if "resultados" in st.session_state:
    df_resultados = st.session_state["resultados"]

    st.markdown("---")
    st.markdown('<p class="section-title">Resultados</p>', unsafe_allow_html=True)

    n_total = len(df_resultados)
    n_solv = (df_resultados["Predicción"].str.contains("Solvente")).sum()
    n_mor = (df_resultados["Predicción"].str.contains("Moroso")).sum()
    pct_solv = n_solv / n_total * 100
    pct_mor = n_mor / n_total * 100
    riesgo_promedio = df_resultados["Prob. Moroso (%)"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-card info"><div class="label">Total</div>'
            f'<div class="value">{n_total}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card success"><div class="label">Solventes</div>'
            f'<div class="value">{n_solv}</div>'
            f'<div class="delta">{pct_solv:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card danger"><div class="label">Morosos</div>'
            f'<div class="value">{n_mor}</div>'
            f'<div class="delta">{pct_mor:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        cls = "success" if riesgo_promedio < 40 else ("warning" if riesgo_promedio < 60 else "danger")
        st.markdown(
            f'<div class="stat-card {cls}"><div class="label">Riesgo prom.</div>'
            f'<div class="value">{riesgo_promedio:.1f}%</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    styled = df_resultados.style.map(colorear_prediccion, subset=["Predicción"])
    st.dataframe(styled, use_container_width=True, height=420, hide_index=True)

    st.markdown("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.download_button(
            label="Descargar resultados (CSV)",
            data=df_resultados.to_csv(index=False).encode("utf-8"),
            file_name=f"resultados_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown('<p class="section-title">Distribución de riesgo</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Por nivel de riesgo**")
        distribucion = df_resultados["Nivel de Riesgo"].value_counts().reindex(
            ["Bajo", "Moderado", "Alto", "Muy Alto"], fill_value=0
        )
        st.bar_chart(distribucion, color=["#16a34a"], height=300)
    with c2:
        st.markdown("**Probabilidad de morosidad**")
        proba = df_resultados["Prob. Moroso (%)"]
        hist = pd.cut(proba, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).value_counts().sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist, color=["#1a365d"], height=300)


# ============================================================
# TABS DE INFO
# ============================================================
st.markdown("---")
st.markdown('<p class="section-title">Información</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Métricas del modelo", "Variables", "Acerca de"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "84.05%")
    c2.metric("AUC-ROC", "0.927")
    c3.metric("Gini", "0.854")
    c4.metric("Test set", "420")
    st.caption("Métricas obtenidas sobre el test set de 420 instancias balanceadas con SMOTE.")

with tab2:
    descripciones = {
        "checking_status": "Estado de cuenta corriente",
        "duration": "Duración del crédito (meses)",
        "credit_history": "Historial crediticio",
        "purpose": "Propósito del crédito",
        "credit_amount": "Monto del crédito",
        "savings_status": "Estado de ahorros",
        "employment": "Antigüedad laboral",
        "installment_commitment": "Tasa de cuota sobre ingreso",
        "personal_status": "Estado civil y sexo",
        "other_parties": "Otros deudores / garantes",
        "residence_since": "Años en residencia actual",
        "property_magnitude": "Tipo de propiedad",
        "age": "Edad (años)",
        "other_payment_plans": "Otros planes de pago",
        "housing": "Tipo de vivienda",
        "existing_credits": "Créditos existentes en el banco",
        "job": "Tipo de empleo",
        "num_dependents": "Número de dependientes",
        "own_telephone": "Teléfono registrado",
        "foreign_worker": "Trabajador extranjero",
    }
    tabla = pd.DataFrame(
        [{"Variable": col, "Descripción": descripciones.get(col, "")} for col in feature_names]
    )
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    with st.expander("Valores posibles de variables categóricas"):
        for col, opciones in categorical_options.items():
            st.markdown(f"**`{col}`**: " + ", ".join(f"`{v}`" for v in opciones))

with tab3:
    st.markdown(
        """
        **Proyecto:** Aprendizaje Estadístico · UPAO

        **Dataset:** German Credit Data (1000 instancias, 20 features, 1 target)
        Originalmente del Prof. Dr. Hans Hofmann (Universität Hamburg).

        **Pipeline de preprocesamiento:**
        1. Carga y limpieza
        2. OrdinalEncoder en 13 variables categóricas
        3. MinMaxScaler en 7 variables numéricas
        4. SMOTE (300 bad → 700 bad sintéticas, 50/50 balance)
        5. Split 70/30 estratificado
        6. Random Forest (100 árboles, class_weight=balanced)

        **Fuente:** [credit-g.arff](https://raw.githubusercontent.com/Waikato/weka-3.8/master/wekadocs/data/credit-g.arff)
        """
    )
