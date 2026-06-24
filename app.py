"""
Credit Scoring - German Credit Data
Aplicación Streamlit profesional con diseño moderno y centrado.
Soporta multiples formatos de CSV (con/sin headers, UCI, latin-1, etc).
"""

import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import arff


# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Credit Scoring | German Credit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Credit Scoring - UPAO - Aprendizaje Estadístico"},
)


# ============================================================
# CSS - DISEÑO LIMPIO, CENTRADO Y ARMONIOSO
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Tipografía global */
    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ocultar elementos de UI innecesarios */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ============================================
       LAYOUT: CONTENIDO CENTRADO Y ARMONIOSO
       ============================================ */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1080px;
    }
    [data-testid="stAppViewContainer"] > .main {
        max-width: 1080px;
        margin: 0 auto;
    }

    /* Sidebar: ancho controlado y contenido protegido */
    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 280px !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        max-width: 100%;
        padding: 1.2rem 0.9rem;
        margin: 0;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        max-width: 100%;
        overflow: hidden;
    }

    /* ============================================
       ANIMACIONES SUAVES
       ============================================ */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stat-card, .hero, .info-card, .success-card {
        animation: fadeInUp 0.4s ease-out;
    }

    /* ============================================
       HERO
       ============================================ */
    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 50%, #1f4e79 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(31, 78, 121, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -60%;
        right: -15%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero h1 {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    .hero p {
        color: rgba(255, 255, 255, 0.92);
        font-size: 0.98rem;
        margin: 0;
        line-height: 1.5;
        max-width: 700px;
        position: relative;
        z-index: 1;
    }
    .hero-badges {
        margin-top: 1.2rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* ============================================
       SECCIONES
       ============================================ */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e3a5f;
        margin: 1.5rem 0 0.4rem 0;
        letter-spacing: -0.01em;
    }
    .section-subtitle {
        color: #6b7785;
        font-size: 0.9rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    /* ============================================
       STAT CARDS
       ============================================ */
    .stat-card {
        background: white;
        padding: 1.1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid #e5e9f0;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
        height: 100%;
    }
    .stat-card:hover {
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    .stat-card .label {
        color: #6b7785;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .stat-card .value {
        color: #1e3a5f;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0;
    }
    .stat-card .delta {
        color: #6b7785;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.2rem;
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
       SIDEBAR METRICS (compactos y protegidos)
       ============================================ */
    .sb-metric {
        background: #f8fafc;
        padding: 0.55rem 0.75rem;
        border-radius: 6px;
        border-left: 3px solid #1f4e79;
        margin-bottom: 0.45rem;
        overflow: hidden;
    }
    .sb-metric .lbl {
        color: #6b7785;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .sb-metric .val {
        color: #1e3a5f;
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 0.1rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .sb-section-title {
        color: #1e3a5f;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0.8rem 0 0.4rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #e5e9f0;
    }
    .sb-brand {
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 0.3rem;
    }
    .sb-brand-title {
        color: #1e3a5f;
        font-size: 1.05rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .sb-brand-sub {
        color: #9ca3af;
        font-size: 0.7rem;
        margin: 0.1rem 0 0 0;
    }

    /* ============================================
       FILE UPLOADER
       ============================================ */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e8eef5 100%);
        border: 2px dashed #1f4e79;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-color: #1e40af;
    }
    [data-testid="stFileUploaderDropzone"] section {
        color: #1f4e79;
    }

    /* ============================================
       BOTONES
       ============================================ */
    .stButton>button {
        background: linear-gradient(135deg, #1f4e79 0%, #2c5282 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        border: none;
        box-shadow: 0 3px 10px rgba(31, 78, 121, 0.25);
        transition: all 0.2s ease;
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
        padding: 0.55rem 1.3rem;
        border: none;
        box-shadow: 0 3px 10px rgba(22, 163, 74, 0.25);
        transition: all 0.2s ease;
    }
    .stDownloadButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 16px rgba(22, 163, 74, 0.4);
    }

    /* ============================================
       TABS
       ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: transparent;
        padding: 0;
        border-bottom: 2px solid #e5e9f0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #6b7785;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #1f4e79 !important;
        border-bottom: 2px solid #1f4e79;
        margin-bottom: -2px;
    }

    /* ============================================
       ALERTAS Y CARDS
       ============================================ */
    .info-card {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #1e40af;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #14532d;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .error-card {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #7f1d1d;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #6b7785;
    }
    .empty-state .icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        opacity: 0.5;
    }
    .empty-state .title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
        margin: 0.3rem 0;
    }
    .empty-state .subtitle {
        font-size: 0.9rem;
        margin: 0;
    }

    /* Code */
    code {
        background: #f0f4f8;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-size: 0.88em;
        color: #1f4e79;
    }

    /* DataFrame tweaks */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ============================================
       RESPONSIVE
       ============================================ */
    @media (max-width: 768px) {
        .hero { padding: 1.5rem 1.2rem; }
        .hero h1 { font-size: 1.5rem; }
        .stat-card .value { font-size: 1.3rem; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }

    /* Scrollbar en sidebar */
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        scrollbar-width: thin;
    }
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
@st.cache_resource(show_spinner="Cargando modelo entrenado...")
def load_models():
    try:
        return (
            joblib.load("preprocessor.pkl"),
            joblib.load("random_forest_model.pkl"),
            joblib.load("feature_names.pkl"),
            joblib.load("categorical_options.pkl"),
        )
    except FileNotFoundError as e:
        st.error(
            f"❌ No se encontró el archivo: **{e.filename}**\n\n"
            "Asegúrate de que los 4 archivos `.pkl` estén junto a `app.py`:\n"
            "- `preprocessor.pkl`\n"
            "- `random_forest_model.pkl`\n"
            "- `feature_names.pkl`\n"
            "- `categorical_options.pkl`"
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()


preprocessor, model, feature_names, categorical_options = load_models()


# ============================================================
# CARGA DE ARCHIVOS - ULTRA ROBUSTA
# ============================================================
COLUMNAS_UCI = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment", "installment_commitment",
    "personal_status", "other_parties", "residence_since", "property_magnitude",
    "age", "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]

# Mapeo de códigos UCI (A11, A12...) a valores legibles
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
    """Detecta y decodifica códigos UCI (A11, A12...) a valores legibles."""
    df = df.copy()
    hubo_cambios = False
    for col, mapeo in MAPEO_CODIGOS_UCI.items():
        if col in df.columns:
            valores_unicos = set(df[col].astype(str).unique())
            codigos_presentes = valores_unicos & set(mapeo.keys())
            if codigos_presentes:
                df[col] = df[col].astype(str).map(lambda x: mapeo.get(x, x))
                hubo_cambios = True

    if "class" in df.columns:
        # 1=good, 2=bad en formato UCI
        valores_class = set(df["class"].astype(str).unique())
        if valores_class & {"1", "2"}:
            df["class"] = df["class"].astype(str).map({"1": "good", "2": "bad"}).fillna(df["class"])
    return df


def decodificar_inteligente(contenido_bytes: bytes) -> str:
    """Prueba varios encodings hasta encontrar uno que funcione."""
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]:
        try:
            return contenido_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return contenido_bytes.decode("latin-1", errors="replace")


def _looks_like_uci_codes(values: list) -> bool:
    """Detecta si los nombres de columna parecen códigos UCI (A11, A12, A30, etc.)"""
    if not values:
        return False
    sample = [str(v) for v in values[:5]]
    n_codes = sum(1 for v in sample if v.startswith("A") and v[1:].isdigit())
    return n_codes >= 2  # Si al menos 2 columnas parecen códigos UCI


def _probar_con_header(texto: str, sep: str) -> pd.DataFrame | None:
    """Intenta parsear ASUMIENDO que TIENE header."""
    try:
        df = pd.read_csv(
            io.StringIO(texto),
            sep=sep,
            skipinitialspace=True,
            skip_blank_lines=True,
            comment="#",
            header=0,
            on_bad_lines="skip",
        )
        if 20 <= len(df.columns) <= 25:
            # Si las columnas parecen códigos UCI (A11, A12, ...), no es header real
            if _looks_like_uci_codes(list(df.columns)):
                return None
            return df
    except Exception:
        return None
    return None


def _probar_sin_header(texto: str, sep: str) -> pd.DataFrame | None:
    """Intenta parsear asumiendo que NO tiene header."""
    try:
        df = pd.read_csv(
            io.StringIO(texto),
            sep=sep,
            skipinitialspace=True,
            skip_blank_lines=True,
            comment="#",
            header=None,
            on_bad_lines="skip",
        )
        if df.shape[1] in (20, 21):
            if df.shape[1] == 21:
                df.columns = COLUMNAS_UCI
            else:
                df.columns = COLUMNAS_UCI[:-1]
            return df
    except Exception:
        return None
    return None


def cargar_csv_robusto(archivo) -> pd.DataFrame:
    """Carga CSV probando múltiples combinaciones de encoding/separador/header."""
    contenido_bytes = archivo.read()
    archivo.seek(0)

    texto = decodificar_inteligente(contenido_bytes)

    separadores = [",", ";", "\t", " ", "|", ":"]

    # Para separadores de espacio/tab, probar SIN header primero (formato UCI)
    for sep in [" ", "\t", "|", ":"]:
        df = _probar_sin_header(texto, sep)
        if df is not None:
            return decodificar_codigos_uci(df)

    # Para coma/punto-y-coma, probar CON header primero (formato normal)
    for sep in [",", ";", "\t", " ", "|", ":"]:
        df = _probar_con_header(texto, sep)
        if df is not None:
            return decodificar_codigos_uci(df)

    # Último intento: sin header con separadores normales
    for sep in [",", ";", " ", "|", ":"]:
        df = _probar_sin_header(texto, sep)
        if df is not None:
            return decodificar_codigos_uci(df)

    raise ValueError(
        "No se pudo leer el archivo. Verifica que sea un CSV/ARFF válido con las 20 columnas requeridas."
    )


def cargar_arff(archivo) -> pd.DataFrame:
    """Carga un ARFF y decodifica bytes."""
    contenido = archivo.read().decode("utf-8")
    datos, _ = arff.loadarff(io.StringIO(contenido))
    df = pd.DataFrame(datos)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode("utf-8")
    return df


def cargar_archivo(archivo) -> pd.DataFrame:
    """Detecta el tipo y carga el archivo."""
    nombre = archivo.name.lower()
    if nombre.endswith(".arff"):
        return cargar_arff(archivo)
    return cargar_csv_robusto(archivo)


def validar_columnas(df: pd.DataFrame) -> list:
    return [c for c in feature_names if c not in df.columns]


def generar_csv_ejemplo() -> bytes:
    """Genera un CSV de muestra en formato estándar (con headers) para que el usuario pruebe."""
    muestra = pd.DataFrame({
        "cliente_id": [f"CLI-{i:04d}" for i in range(1, 6)],
        "nombre": ["Juan Pérez", "María López", "Carlos Ruiz", "Ana Torres", "Luis Vega"],
        "checking_status": ["no checking", "<0", "0<=X<200", ">=200", "no checking"],
        "duration": [24, 36, 12, 48, 18],
        "credit_history": ["existing paid", "critical/other existing credit", "all paid", "existing paid", "delayed previously"],
        "purpose": ["radio/tv", "new car", "furniture/equipment", "business", "education"],
        "credit_amount": [2500, 5000, 1500, 8000, 3000],
        "savings_status": ["<100", "<100", "100<=X<500", ">=1000", "<100"],
        "employment": ["1<=X<4", ">=7", "<1", ">=7", "1<=X<4"],
        "installment_commitment": [2, 3, 2, 4, 2],
        "personal_status": ["male single", "male single", "female div/dep/mar", "male single", "male mar/wid"],
        "other_parties": ["none", "none", "none", "co applicant", "none"],
        "residence_since": [2, 4, 1, 3, 2],
        "property_magnitude": ["car", "real estate", "life insurance", "real estate", "no known property"],
        "age": [35, 45, 28, 52, 31],
        "other_payment_plans": ["none", "bank", "none", "none", "stores"],
        "housing": ["own", "own", "rent", "own", "for free"],
        "existing_credits": [1, 2, 1, 3, 1],
        "job": ["skilled", "high qualif/self emp/mgmt", "unskilled resident", "skilled", "unskilled resident"],
        "num_dependents": [1, 1, 1, 2, 1],
        "own_telephone": ["yes", "none", "yes", "yes", "none"],
        "foreign_worker": ["yes", "yes", "yes", "no", "yes"],
    })
    return muestra.to_csv(index=False).encode("utf-8")


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
        '<div class="sb-brand">'
        '<p class="sb-brand-title">🏦 Credit Scoring</p>'
        '<p class="sb-brand-sub">UPAO · Aprendizaje Estadístico</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown('<p class="sb-section-title">📊 Modelo</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-metric"><div class="lbl">Accuracy</div><div class="val">84.05%</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-metric"><div class="lbl">AUC-ROC</div><div class="val">0.927</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-metric"><div class="lbl">Gini</div><div class="val">0.854</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-metric"><div class="lbl">Test set</div><div class="val">420 clientes</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="sb-section-title">🧠 Pipeline</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #4a5568; line-height: 1.8; padding-left: 0.2rem;">
        • MinMaxScaler (numéricas)<br>
        • OrdinalEncoder (categóricas)<br>
        • SMOTE (50/50 balanceo)<br>
        • Random Forest (100 árboles)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size: 0.7rem; color: #9ca3af; text-align: center; padding: 0.3rem 0;">v1.0 · WEKA replication</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🏦 Credit Scoring</h1>
        <p>Evalúa el riesgo crediticio de tus clientes con un modelo de Machine Learning entrenado con el German Credit Data.</p>
        <div class="hero-badges">
            <span class="hero-badge">⚡ Predicción por lotes</span>
            <span class="hero-badge">📂 CSV / ARFF</span>
            <span class="hero-badge">📥 Descarga de resultados</span>
            <span class="hero-badge">🔍 Múltiples formatos</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ZONA DE CARGA
# ============================================================
st.markdown('<p class="section-title">📂 Cargar archivo de clientes</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">Sube un archivo con los datos de tus clientes. '
    "Acepta CSV con o sin encabezados, formato UCI, ARFF, y múltiples encodings.</p>",
    unsafe_allow_html=True,
)

col_upload, col_sample = st.columns([3, 1])
with col_upload:
    archivo_subido = st.file_uploader(
        "Arrastra o selecciona tu archivo",
        type=["csv", "arff", "txt", "data"],
        help="Formatos soportados: CSV (con o sin headers), ARFF, texto plano, formato UCI.",
        label_visibility="collapsed",
    )
with col_sample:
    st.markdown('<p style="font-size: 0.8rem; color: #6b7785; margin: 0.5rem 0 0.4rem 0; text-align: center;">¿No tienes archivo?</p>', unsafe_allow_html=True)
    st.download_button(
        "📋 Descargar muestra",
        data=generar_csv_ejemplo(),
        file_name="muestra_clientes.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Empty state cuando no hay archivo
if archivo_subido is None and "resultados" not in st.session_state:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">📊</div>
            <p class="title">Esperando tu archivo</p>
            <p class="subtitle">Sube un CSV/ARFF o descarga la muestra para probar</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Procesar archivo
if archivo_subido is not None:
    try:
        with st.spinner("📖 Leyendo archivo..."):
            df_input = cargar_archivo(archivo_subido)
    except Exception as e:
        st.markdown(
            f'<div class="error-card">'
            f"❌ <b>No se pudo leer el archivo</b><br><br>"
            f"<b>Detalle:</b> {e}<br><br>"
            f"💡 <b>Formatos soportados:</b><br>"
            f"• CSV con headers descriptivos (checking_status, duration, etc.)<br>"
            f"• CSV sin headers en formato UCI (21 columnas, códigos A11, A12, ...)<br>"
            f"• ARFF estándar de WEKA<br>"
            f"• Texto separado por coma, punto-y-coma, tab o espacio<br><br>"
            f"📋 Descarga la muestra CSV de arriba para ver el formato esperado."
            f"</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    if "class" in df_input.columns:
        st.markdown(
            '<div class="info-card">ℹ️ Se detectó la columna <code>class</code> en el archivo. '
            "Será ignorada para la predicción.</div>",
            unsafe_allow_html=True,
        )
        df_input = df_input.drop(columns=["class"])

    faltantes = validar_columnas(df_input)
    if faltantes:
        st.markdown(
            f'<div class="info-card" style="border-color: #fecaca; background: #fef2f2; color: #7f1d1d;">'
            f"❌ <b>Faltan {len(faltantes)} columnas</b> requeridas por el modelo:<br><br>"
            + ", ".join(f"<code>{c}</code>" for c in faltantes)
            + "</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Ver las 20 columnas esperadas"):
            st.write(", ".join(f"`{c}`" for c in feature_names))
    else:
        st.markdown(
            f'<div class="success-card">✅ <b>Archivo cargado correctamente</b> · '
            f"{len(df_input)} clientes detectados</div>",
            unsafe_allow_html=True,
        )

        with st.expander(f"👀 Vista previa (primeras 10 filas de {len(df_input)})", expanded=False):
            st.dataframe(df_input.head(10), use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Analizar Riesgo Crediticio", type="primary", use_container_width=True):
                with st.spinner("🧠 Procesando predicciones..."):
                    try:
                        df_resultados = predecir_lote(df_input)
                        st.session_state["resultados"] = df_resultados
                    except Exception as e:
                        st.error(f"❌ Error al predecir: {e}")
                        st.stop()


# ============================================================
# RESULTADOS
# ============================================================
if "resultados" in st.session_state:
    df_resultados = st.session_state["resultados"]

    st.markdown("---")
    st.markdown('<p class="section-title">📊 Resultados del análisis</p>', unsafe_allow_html=True)

    n_total = len(df_resultados)
    n_solv = (df_resultados["Predicción"].str.contains("Solvente")).sum()
    n_mor = (df_resultados["Predicción"].str.contains("Moroso")).sum()
    pct_solv = n_solv / n_total * 100
    pct_mor = n_mor / n_total * 100
    riesgo_promedio = df_resultados["Prob. Moroso (%)"].mean()

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-card info"><div class="label">Total Clientes</div>'
            f'<div class="value">{n_total}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card success"><div class="label">Solventes</div>'
            f'<div class="value">{n_solv}</div>'
            f'<div class="delta">{pct_solv:.1f}% del total</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card danger"><div class="label">Morosos</div>'
            f'<div class="value">{n_mor}</div>'
            f'<div class="delta">{pct_mor:.1f}% del total</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        color_class = "success" if riesgo_promedio < 40 else ("warning" if riesgo_promedio < 60 else "danger")
        st.markdown(
            f'<div class="stat-card {color_class}"><div class="label">Riesgo Promedio</div>'
            f'<div class="value">{riesgo_promedio:.1f}%</div>'
            f'<div class="delta">de morosidad</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<p class="section-title">📋 Detalle por cliente</p>', unsafe_allow_html=True)

    styled = df_resultados.style.map(colorear_prediccion, subset=["Predicción"])
    st.dataframe(styled, use_container_width=True, height=420, hide_index=True)

    st.markdown("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.download_button(
            label="📥 Descargar resultados como CSV",
            data=df_resultados.to_csv(index=False).encode("utf-8"),
            file_name=f"resultados_credit_scoring_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Distribución
    st.markdown("---")
    st.markdown('<p class="section-title">📈 Distribución de riesgo</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Clientes por nivel de riesgo**")
        distribucion = df_resultados["Nivel de Riesgo"].value_counts().reindex(
            ["Bajo", "Moderado", "Alto", "Muy Alto"], fill_value=0
        )
        st.bar_chart(distribucion, color=["#15803d"], height=300)
    with c2:
        st.markdown("**Probabilidad de morosidad (distribución)**")
        proba = df_resultados["Prob. Moroso (%)"]
        hist = pd.cut(proba, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).value_counts().sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist, color=["#1f4e79"], height=300)


# ============================================================
# TABS DE INFORMACIÓN
# ============================================================
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Sobre el Modelo", "📋 Variables", "ℹ️ Acerca de"])

with tab1:
    st.markdown("#### Métricas del modelo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "84.05%")
    c2.metric("AUC-ROC", "0.927")
    c3.metric("Gini", "0.854")
    c4.metric("Test set", "420")
    st.caption("Métricas obtenidas sobre el test set de 420 instancias balanceadas con SMOTE.")

    st.markdown("")
    st.markdown("#### Pipeline de preprocesamiento")
    st.markdown(
        """
        | # | Paso | Descripción |
        |---|---|---|
        | 1 | **Carga** | Dataset German Credit (1000 instancias) |
        | 2 | **Encoding** | OrdinalEncoder en 13 variables categóricas |
        | 3 | **Normalización** | MinMaxScaler en 7 variables numéricas |
        | 4 | **Balanceo** | SMOTE (300 bad → 700 bad sintéticas) |
        | 5 | **Split** | 70% training (980) / 30% test (420) estratificado |
        | 6 | **Modelo** | Random Forest (100 árboles, `class_weight=balanced`) |
        """
    )

with tab2:
    st.markdown("#### Las 20 variables del modelo")
    descripciones = {
        "checking_status": ("Categórica", "Estado de la cuenta corriente"),
        "duration": ("Numérica", "Duración del crédito (meses)"),
        "credit_history": ("Categórica", "Historial crediticio"),
        "purpose": ("Categórica", "Propósito del crédito"),
        "credit_amount": ("Numérica", "Monto del crédito"),
        "savings_status": ("Categórica", "Estado de cuenta de ahorros"),
        "employment": ("Categórica", "Antigüedad laboral"),
        "installment_commitment": ("Numérica", "Tasa de cuota sobre ingreso"),
        "personal_status": ("Categórica", "Estado civil y sexo"),
        "other_parties": ("Categórica", "Otros deudores / garantes"),
        "residence_since": ("Numérica", "Años en residencia actual"),
        "property_magnitude": ("Categórica", "Tipo de propiedad"),
        "age": ("Numérica", "Edad en años"),
        "other_payment_plans": ("Categórica", "Otros planes de pago"),
        "housing": ("Categórica", "Tipo de vivienda"),
        "existing_credits": ("Numérica", "Créditos existentes en el banco"),
        "job": ("Categórica", "Tipo de empleo"),
        "num_dependents": ("Numérica", "Número de dependientes"),
        "own_telephone": ("Categórica", "Teléfono registrado"),
        "foreign_worker": ("Categórica", "Trabajador extranjero"),
    }
    tabla = pd.DataFrame(
        [
            {"Variable": col, "Tipo": descripciones[col][0], "Descripción": descripciones[col][1]}
            for col in feature_names
        ]
    )
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    with st.expander("Ver valores posibles de las variables categóricas"):
        for col, opciones in categorical_options.items():
            st.markdown(f"**`{col}`**")
            st.write(", ".join(f"`{v}`" for v in opciones))
            st.markdown("")

with tab3:
    st.markdown(
        """
        #### 🎓 Acerca del proyecto

        - **Curso:** Aprendizaje Estadístico
        - **Universidad:** UPAO
        - **Modelo de referencia:** Random Forest entrenado en WEKA

        #### 📊 Sobre el dataset

        El **German Credit Data** es un dataset clásico de clasificación binaria
        para credit scoring. Contiene 1000 solicitudes de crédito en Alemania con
        20 atributos predictivos y 1 variable objetivo.

        #### 🔄 Notas metodológicas

        1. **SMOTE antes del split:** se replica el flujo de WEKA. Lo académicamente
           correcto sería aplicar SMOTE solo en training.
        2. **Doble balanceo:** SMOTE + `class_weight='balanced'`. Refuerza la
           atención a la clase minoritaria.
        3. **Test set no representativo del mundo real:** tras SMOTE, el test
           queda 50/50. Las métricas son optimistas vs producción.
        4. **Validación contra WEKA:** variaciones de ±2-3% entre Python y WEKA
           son normales.

        #### 🔗 Fuentes

        - WEKA: [credit-g.arff](https://raw.githubusercontent.com/Waikato/weka-3.8/master/wekadocs/data/credit-g.arff)
        - UCI: [Statlog German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
        """
    )
