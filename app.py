"""
Credit Scoring - German Credit Data
Aplicacion Streamlit con selector inteligente de columnas.
"""

import io
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
    initial_sidebar_state="expanded",
    menu_items={"About": "Credit Scoring - UPAO"},
)


# ============================================================
# CSS - LIMPIO Y MINIMALISTA
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Layout centrado */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1080px;
    }
    [data-testid="stAppViewContainer"] > .main {
        max-width: 1080px;
        margin: 0 auto;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        width: 270px !important;
        min-width: 270px !important;
        max-width: 270px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 270px !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        max-width: 100%;
        padding: 1.2rem 0.9rem;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        max-width: 100%;
        overflow: hidden;
    }

    /* Header minimalista */
    .app-header {
        padding: 1.2rem 0 1.5rem 0;
        border-bottom: 1px solid #e5e9f0;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1e3a5f;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        font-size: 0.88rem;
        color: #6b7785;
        margin: 0.2rem 0 0 0;
    }

    /* Section titles */
    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1e3a5f;
        margin: 1.5rem 0 0.6rem 0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Mode selector cards */
    .mode-card {
        background: white;
        border: 1px solid #e5e9f0;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        height: 100%;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .mode-card.active {
        border-color: #1f4e79;
        background: #eff6ff;
        box-shadow: 0 2px 8px rgba(31, 78, 121, 0.1);
    }
    .mode-card .mode-title {
        font-weight: 700;
        color: #1e3a5f;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }
    .mode-card .mode-desc {
        font-size: 0.78rem;
        color: #6b7785;
        line-height: 1.4;
    }

    /* Stat cards */
    .stat-card {
        background: white;
        padding: 1rem 1.1rem;
        border-radius: 8px;
        border: 1px solid #e5e9f0;
        transition: all 0.2s ease;
        height: 100%;
    }
    .stat-card:hover {
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }
    .stat-card .label {
        color: #6b7785;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0.2rem 0 0 0;
        color: #1e3a5f;
    }
    .stat-card .delta {
        color: #6b7785;
        font-size: 0.8rem;
        margin-top: 0.1rem;
    }
    .stat-card.success { border-left: 3px solid #16a34a; }
    .stat-card.success .value { color: #15803d; }
    .stat-card.danger { border-left: 3px solid #dc2626; }
    .stat-card.danger .value { color: #b91c1c; }
    .stat-card.warning { border-left: 3px solid #f59e0b; }
    .stat-card.warning .value { color: #b45309; }
    .stat-card.info { border-left: 3px solid #3b82f6; }
    .stat-card.info .value { color: #1e40af; }

    /* Sidebar metrics */
    .sb-metric {
        background: #f8fafc;
        padding: 0.5rem 0.7rem;
        border-radius: 6px;
        border-left: 3px solid #1f4e79;
        margin-bottom: 0.4rem;
        overflow: hidden;
    }
    .sb-metric .lbl {
        color: #6b7785;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .sb-metric .val {
        color: #1e3a5f;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .sb-section-title {
        color: #1e3a5f;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0.8rem 0 0.4rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #e5e9f0;
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e8eef5 100%);
        border: 2px dashed #1f4e79;
        border-radius: 10px;
        padding: 1.5rem;
    }

    /* Botones */
    .stButton>button {
        background: linear-gradient(135deg, #1f4e79 0%, #2c5282 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        border: none;
        box-shadow: 0 3px 10px rgba(31, 78, 121, 0.25);
    }
    .stButton>button:hover {
        box-shadow: 0 5px 16px rgba(31, 78, 121, 0.4);
        transform: translateY(-1px);
    }
    .stDownloadButton>button {
        background: linear-gradient(135deg, #15803d 0%, #16a34a 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        border: none;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
        border-bottom: 2px solid #e5e9f0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px 6px 0 0;
        padding: 0.4rem 0.9rem;
        font-weight: 500;
        color: #6b7785;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #1f4e79 !important;
        border-bottom: 2px solid #1f4e79;
        margin-bottom: -2px;
    }

    /* Alerts */
    .info-card {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        color: #1e40af;
        font-size: 0.88rem;
        margin: 0.4rem 0;
    }
    .success-card {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        color: #14532d;
        font-size: 0.88rem;
        margin: 0.4rem 0;
    }
    .error-card {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        color: #7f1d1d;
        font-size: 0.88rem;
        margin: 0.4rem 0;
    }
    .warning-card {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        color: #78350f;
        font-size: 0.88rem;
        margin: 0.4rem 0;
    }

    code {
        background: #f0f4f8;
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
        font-size: 0.85em;
        color: #1f4e79;
    }

    /* Scrollbar */
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar {
        width: 4px;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 2px;
    }
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
            "Necesitas los 5 archivos `.pkl` junto a `app.py`."
        )
        st.stop()


preprocessor, model, feature_names, categorical_options, feature_defaults = load_models()


# ============================================================
# MODOS DE COLUMNAS
# ============================================================
# Mapeo de nombres amigables a nombres internos del modelo
NOMBRES_AMIGABLES = {
    "Age": "age",
    "Sex": "personal_status",
    "Job": "job",
    "Housing": "housing",
    "Saving accounts": "savings_status",
    "Checking account": "checking_status",
    "Credit amount": "credit_amount",
    "Duration": "duration",
    "Purpose": "purpose",
}

# Columnas que el usuario puede elegir
COLUMNAS_SIMPLIFICADAS = list(NOMBRES_AMIGABLES.keys())


# ============================================================
# UTILIDADES DE CARGA
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

    raise ValueError("No se pudo parsear el archivo.")


def normalizar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea nombres amigables a nombres internos del modelo."""
    df = df.copy()
    rename = {}
    for col_amigable, col_interna in NOMBRES_AMIGABLES.items():
        if col_amigable in df.columns and col_interna not in df.columns:
            rename[col_amigable] = col_interna
    if rename:
        df = df.rename(columns=rename)
    return df


def aplicar_defaults(df: pd.DataFrame, columnas_presentes: list) -> pd.DataFrame:
    """Llena las features faltantes con sus valores por defecto del training set."""
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
        return "background-color: #dcfce7; color: #14532d; font-weight: 600"
    if "Moroso" in str(val):
        return "background-color: #fee2e2; color: #7f1d1d; font-weight: 600"
    return ""


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding:0.3rem 0 0.5rem 0;">'
        '<div style="font-size:1.4rem; line-height:1;">🏦</div>'
        '<div style="font-weight:800; color:#1e3a5f; font-size:1rem; margin-top:0.2rem;">Credit Scoring</div>'
        '<div style="color:#9ca3af; font-size:0.7rem; margin-top:0.1rem;">UPAO</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown('<p class="sb-section-title">Modelo</p>', unsafe_allow_html=True)
    st.markdown('<div class="sb-metric"><div class="lbl">Accuracy</div><div class="val">84.05%</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-metric"><div class="lbl">AUC-ROC</div><div class="val">0.927</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-metric"><div class="lbl">Gini</div><div class="val">0.854</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-metric"><div class="lbl">Test set</div><div class="val">420 clientes</div></div>', unsafe_allow_html=True)

    st.markdown('<p class="sb-section-title">Pipeline</p>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.75rem; color:#4a5568; line-height:1.7;">'
        "MinMaxScaler · OrdinalEncoder<br>SMOTE (50/50)<br>Random Forest (100 árboles)"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.68rem; color:#9ca3af; text-align:center;">v1.1</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER MINIMALISTA
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
# SELECTOR DE MODO DE COLUMNAS
# ============================================================
st.markdown('<p class="section-title">Modo de análisis</p>', unsafe_allow_html=True)

modo = st.radio(
    "Selecciona qué columnas tendrá tu archivo",
    options=[
        "Completo (20 columnas)",
        "Simplificado (9 columnas)",
        "Personalizado",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# Mostrar info del modo seleccionado
if modo == "Completo (20 columnas)":
    st.markdown(
        '<div class="info-card">📋 El archivo debe contener las <b>20 columnas</b> completas del modelo '
        "(checking_status, duration, credit_history, purpose, credit_amount, savings_status, employment, "
        "installment_commitment, personal_status, other_parties, residence_since, property_magnitude, age, "
        "other_payment_plans, housing, existing_credits, job, num_dependents, own_telephone, foreign_worker).</div>",
        unsafe_allow_html=True,
    )
    columnas_esperadas = feature_names
    requiere_personalizar = False
elif modo == "Simplificado (9 columnas)":
    st.markdown(
        f'<div class="info-card">📋 El archivo debe contener solo estas <b>9 columnas</b>: '
        + ", ".join(f"<code>{c}</code>" for c in COLUMNAS_SIMPLIFICADAS)
        + ". Las 11 restantes se completan automáticamente con valores típicos del dataset.</div>",
        unsafe_allow_html=True,
    )
    columnas_esperadas = [NOMBRES_AMIGABLES[c] for c in COLUMNAS_SIMPLIFICADAS]
    requiere_personalizar = False
else:  # Personalizado
    cols_seleccionadas = st.multiselect(
        "Selecciona las columnas que tendrá tu archivo",
        options=feature_names,
        default=feature_names,
        help="Las columnas no seleccionadas se completan con valores por defecto del training set",
    )
    columnas_esperadas = cols_seleccionadas
    requiere_personalizar = True
    if len(cols_seleccionadas) == 0:
        st.markdown(
            '<div class="warning-card">⚠️ Selecciona al menos una columna.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="info-card">📋 Tu archivo tendrá <b>{len(cols_seleccionadas)} columnas</b>. '
            f"Las {20 - len(cols_seleccionadas)} restantes se completan automáticamente.</div>",
            unsafe_allow_html=True,
        )

# ============================================================
# UPLOAD
# ============================================================
st.markdown('<p class="section-title">Cargar archivo</p>', unsafe_allow_html=True)

archivo_subido = st.file_uploader(
    "Arrastra o selecciona tu archivo CSV/ARFF",
    type=["csv", "arff", "txt", "data"],
    help="Formatos soportados: CSV (con o sin headers, formato UCI), ARFF.",
    label_visibility="collapsed",
)

# Empty state
if archivo_subido is None and "resultados" not in st.session_state:
    st.markdown(
        '<div style="text-align:center; padding:2rem 1rem; color:#9ca3af; font-size:0.9rem;">'
        "Esperando archivo...</div>",
        unsafe_allow_html=True,
    )

if archivo_subido is not None and len(columnas_esperadas) > 0:
    try:
        with st.spinner("Leyendo archivo..."):
            df_input = cargar_archivo_robusto(archivo_subido)
    except Exception as e:
        st.markdown(
            f'<div class="error-card">❌ <b>No se pudo leer el archivo</b><br><br>{e}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # Normalizar nombres (amigables -> internos)
    df_input = normalizar_nombres_columnas(df_input)

    # Quitar columna target si existe
    if "class" in df_input.columns:
        df_input = df_input.drop(columns=["class"])

    # Verificar columnas presentes
    presentes = [c for c in columnas_esperadas if c in df_input.columns]
    faltantes = [c for c in columnas_esperadas if c not in df_input.columns]

    if not presentes:
        st.markdown(
            f'<div class="error-card">❌ Ninguna de las columnas esperadas está en el archivo.<br><br>'
            f"<b>Esperadas ({len(columnas_esperadas)}):</b> "
            + ", ".join(f"<code>{c}</code>" for c in columnas_esperadas)
            + "</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    # Si modo completo, requerir TODAS las 20
    if modo == "Completo (20 columnas)" and len(presentes) < 20:
        st.markdown(
            f'<div class="error-card">❌ Modo completo requiere las <b>20 columnas</b>.<br><br>'
            f"<b>Faltan {len(faltantes)}:</b> "
            + ", ".join(f"<code>{c}</code>" for c in faltantes[:5])
            + ("..." if len(faltantes) > 5 else "")
            + '<br><br>💡 Cambia a modo <b>"Simplificado"</b> o <b>"Personalizado"</b> '
            "si tu archivo no tiene todas las columnas.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    # Aplicar defaults a las columnas faltantes
    df_completo, defaults_usados = aplicar_defaults(df_input, presentes)

    if defaults_usados:
        st.markdown(
            f'<div class="info-card">ℹ️ <b>{len(defaults_usados)} columnas</b> completadas automáticamente '
            f"con valores por defecto del training set: "
            + ", ".join(f"<code>{c}</code>" for c in defaults_usados[:4])
            + ("..." if len(defaults_usados) > 4 else "")
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="success-card">✅ <b>Archivo listo</b> · {len(df_input)} clientes · '
        f"{len(presentes)} columnas del usuario + {len(defaults_usados)} por defecto</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Vista previa (primeras 5 filas)", expanded=False):
        st.dataframe(df_input.head(5), use_container_width=True, hide_index=True)

    # Botón de predicción
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Predecir", type="primary", use_container_width=True):
            with st.spinner("Procesando..."):
                try:
                    st.session_state["resultados"] = predecir_lote(df_completo)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
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
        distribucion = df_resultados["Nivel de Riesgo"].value_counts().reindex(
            ["Bajo", "Moderado", "Alto", "Muy Alto"], fill_value=0
        )
        st.bar_chart(distribucion, color=["#15803d"], height=280)
    with c2:
        proba = df_resultados["Prob. Moroso (%)"]
        hist = pd.cut(proba, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).value_counts().sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist, color=["#1f4e79"], height=280)


# ============================================================
# TABS DE INFORMACIÓN
# ============================================================
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Métricas", "Variables", "Acerca de"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "84.05%")
    c2.metric("AUC-ROC", "0.927")
    c3.metric("Gini", "0.854")
    c4.metric("Test", "420")
    st.caption("Métricas sobre test set balanceado con SMOTE.")

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
        [
            {"Variable": col, "Descripción": descripciones.get(col, "")}
            for col in feature_names
        ]
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
        Originalmente de Prof. Dr. Hans Hofmann (Universität Hamburg).

        **Pipeline:** MinMaxScaler → OrdinalEncoder → SMOTE (50/50) → Random Forest (100 árboles)

        **Fuente:** [credit-g.arff](https://raw.githubusercontent.com/Waikato/weka-3.8/master/wekadocs/data/credit-g.arff)
        """
    )
