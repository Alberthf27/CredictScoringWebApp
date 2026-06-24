"""
Credit Scoring - German Credit Data
Aplicación Streamlit profesional con diseño moderno.
Acepta archivos CSV/ARFF en múltiples formatos (con/sin headers, UCI, etc.).
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
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Credit Scoring App - UPAO - Aprendizaje Estadístico"
    }
)


# ============================================================
# CSS - DISEÑO LIMPIO Y PROFESIONAL
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ocultar elementos de UI innecesarios */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Hero principal */
    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 50%, #1f4e79 100%);
        padding: 3rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(31, 78, 121, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 50%;
    }
    .hero h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    .hero p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    .hero-badges {
        margin-top: 1.5rem;
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
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e9f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    .stat-card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }
    .stat-card .label {
        color: #6b7785;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-card .value {
        color: #1e3a5f;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0 0 0;
    }
    .stat-card.success .value { color: #15803d; }
    .stat-card.danger .value { color: #b91c1c; }
    .stat-card.warning .value { color: #b45309; }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8eef5 100%);
        border: 2px dashed #1f4e79;
        border-radius: 12px;
        padding: 2rem;
    }

    /* Botones */
    .stButton>button {
        background: linear-gradient(135deg, #1f4e79 0%, #2c5282 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 12px rgba(31, 78, 121, 0.25);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(31, 78, 121, 0.4);
        transform: translateY(-1px);
    }
    .stDownloadButton>button {
        background: linear-gradient(135deg, #15803d 0%, #16a34a 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        border: none;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.25);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f0f4f8;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #1f4e79 !important;
        color: white !important;
    }

    /* Risk badge en tabla */
    .risk-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .risk-badge.low { background: #dcfce7; color: #14532d; }
    .risk-badge.high { background: #fee2e2; color: #7f1d1d; }

    /* Section titles */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a5f;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: -0.01em;
    }
    .section-subtitle {
        color: #6b7785;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    /* Code blocks */
    code {
        background: #f0f4f8;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-size: 0.9em;
        color: #1f4e79;
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
# FUNCIONES DE CARGA DE ARCHIVOS (MÚLTIPLES FORMATOS)
# ============================================================
COLUMNAS_UCI = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment", "installment_commitment",
    "personal_status", "other_parties", "residence_since", "property_magnitude",
    "age", "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]


def _leer_csv_inteligente(contenido_bytes: bytes) -> pd.DataFrame:
    """Lee un CSV probando diferentes separadores y encodings."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    separadores = [",", ";", "\t", " "]

    for enc in encodings:
        try:
            texto = contenido_bytes.decode(enc)
        except UnicodeDecodeError:
            continue

        for sep in separadores:
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(texto), sep=sep)

                if len(df.columns) >= 20:
                    return df
            except Exception:
                continue

    raise ValueError("No se pudo leer el archivo CSV con ningún encoding/separador conocido")


def cargar_csv_inteligente(archivo) -> pd.DataFrame:
    """
    Carga un CSV en cualquier formato:
    1. CSV con headers y nombres de columnas
    2. CSV sin headers (formato UCI german.data, 21 columnas)
    3. CSV con codificación latin-1 o BOM UTF-8
    """
    contenido_bytes = archivo.read()
    archivo.seek(0)

    df = _leer_csv_inteligente(contenido_bytes)

    if len(df.columns) == 21 and list(df.columns) == list(range(21)):
        df.columns = COLUMNAS_UCI
    elif len(df.columns) == 21 and all(isinstance(c, int) for c in df.columns):
        df.columns = COLUMNAS_UCI
    elif "class" not in df.columns and "checking_status" in df.columns:
        pass

    return df


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
    return cargar_csv_inteligente(archivo)


def validar_columnas(df: pd.DataFrame) -> list:
    return [c for c in feature_names if c not in df.columns]


# ============================================================
# FUNCIONES DE PREDICCIÓN
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
    st.markdown("### 🏦 Credit Scoring")
    st.caption("UPAO · Aprendizaje Estadístico")
    st.markdown("---")
    st.markdown("#### 📊 Rendimiento del modelo")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", "84.05%", help="Porcentaje de predicciones correctas")
    c2.metric("AUC-ROC", "0.927", help="Área bajo la curva ROC")
    c1.metric("Gini", "0.854", help="2 × AUC − 1")
    c2.metric("Test set", "420", help="Instancias en test (post-SMOTE)")
    st.markdown("---")
    st.markdown("#### 🧠 Pipeline")
    st.markdown(
        """
        - **Normalización:** MinMaxScaler
        - **Encoding:** OrdinalEncoder
        - **Balanceo:** SMOTE (50/50)
        - **Modelo:** Random Forest (100 árboles)
        """
    )
    st.markdown("---")
    st.caption("v1.0 · Replicado desde WEKA")


# ============================================================
# HERO / HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🏦 Credit Scoring</h1>
        <p>Evalúa el riesgo crediticio de tus clientes con un modelo de Machine Learning entrenado con el dataset German Credit Data.</p>
        <div class="hero-badges">
            <span class="hero-badge">⚡ Predicción por lotes</span>
            <span class="hero-badge">📂 CSV / ARFF</span>
            <span class="hero-badge">📥 Descarga de resultados</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD DE ARCHIVO
# ============================================================
st.markdown('<p class="section-title">📂 Cargar archivo de clientes</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">Sube un archivo <code>CSV</code> o <code>ARFF</code> con los datos de los clientes. '
    "Se aceptan archivos con o sin encabezados, en formato estándar o UCI (german.data).</p>",
    unsafe_allow_html=True,
)

archivo_subido = st.file_uploader(
    "Arrastra o selecciona tu archivo",
    type=["csv", "arff", "txt", "data"],
    help="Formatos soportados: CSV (con o sin headers), ARFF, texto plano.",
    label_visibility="collapsed",
)

if archivo_subido is not None:
    try:
        with st.spinner("Leyendo archivo..."):
            df_input = cargar_archivo(archivo_subido)
    except Exception as e:
        st.error(f"❌ No se pudo leer el archivo: {e}")
        st.info(
            "💡 **Tip:** Si tu archivo no tiene headers, asegúrate de que tenga **21 columnas** "
            "(20 features + 1 de clase) en el orden del dataset UCI German Credit."
        )
        st.stop()

    if "class" in df_input.columns:
        st.info("ℹ️ Se detectó la columna `class` en el archivo. Será ignorada para la predicción.")
        df_input = df_input.drop(columns=["class"])

    faltantes = validar_columnas(df_input)
    if faltantes:
        st.error(
            f"❌ Faltan **{len(faltantes)} columnas** requeridas por el modelo:\n\n"
            + ", ".join(f"`{c}`" for c in faltantes)
        )
        st.info(
            "**Columnas esperadas (20):**\n\n"
            + ", ".join(f"`{c}`" for c in feature_names)
        )
    else:
        # Estado: archivo válido
        st.success(f"✅ Archivo cargado correctamente · **{len(df_input)} clientes** detectados")

        with st.expander("👀 Vista previa de los datos", expanded=False):
            st.dataframe(df_input.head(10), use_container_width=True, hide_index=True)

        # ========================================================
        # BOTÓN DE PREDICCIÓN
        # ========================================================
        if st.button("🚀 Analizar Riesgo Crediticio", type="primary", use_container_width=True):
            with st.spinner("Procesando predicciones..."):
                try:
                    df_resultados = predecir_lote(df_input)
                except Exception as e:
                    st.error(f"❌ Error al predecir: {e}")
                    st.stop()

            # Guardar en session state para mantener resultados
            st.session_state["resultados"] = df_resultados

# ============================================================
# RESULTADOS (si ya se predijo)
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

    # ====================================================
    # TARJETAS DE RESUMEN
    # ====================================================
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="label">Total Clientes</div>'
            f'<div class="value">{n_total}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card success"><div class="label">Solventes</div>'
            f'<div class="value">{n_solv} <span style="font-size:1rem;color:#6b7785;">({pct_solv:.1f}%)</span></div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card danger"><div class="label">Morosos</div>'
            f'<div class="value">{n_mor} <span style="font-size:1rem;color:#6b7785;">({pct_mor:.1f}%)</span></div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        color_class = "success" if riesgo_promedio < 40 else ("warning" if riesgo_promedio < 60 else "danger")
        st.markdown(
            f'<div class="stat-card {color_class}"><div class="label">Riesgo Promedio</div>'
            f'<div class="value">{riesgo_promedio:.1f}%</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ====================================================
    # TABLA DE RESULTADOS
    # ====================================================
    st.markdown('<p class="section-title">📋 Detalle por cliente</p>', unsafe_allow_html=True)

    styled = df_resultados.style.map(colorear_prediccion, subset=["Predicción"])
    st.dataframe(styled, use_container_width=True, height=420, hide_index=True)

    # ====================================================
    # DESCARGA
    # ====================================================
    csv_bytes = df_resultados.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar resultados (CSV)",
        data=csv_bytes,
        file_name=f"resultados_credit_scoring_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ====================================================
    # DISTRIBUCIÓN DE RIESGO
    # ====================================================
    st.markdown("---")
    st.markdown('<p class="section-title">📈 Distribución de riesgo</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**Clientes por nivel de riesgo**")
        distribucion = df_resultados["Nivel de Riesgo"].value_counts().reindex(
            ["Bajo", "Moderado", "Alto", "Muy Alto"], fill_value=0
        )
        st.bar_chart(distribucion, color=["#15803d"])

    with c2:
        st.markdown("**Probabilidad de morosidad (histograma)**")
        hist_data = pd.DataFrame(
            {
                "Prob. Moroso (%)": df_resultados["Prob. Moroso (%)"],
            }
        )
        st.bar_chart(
            hist_data["Prob. Moroso (%)"].value_counts(bins=10).sort_index(),
            color=["#1f4e79"],
        )


# ============================================================
# TABS DE INFORMACIÓN
# ============================================================
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Sobre el Modelo", "📋 Variables", "ℹ️ Acerca de"])

with tab1:
    st.markdown("#### Métricas del modelo Random Forest")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "84.05%")
    c2.metric("AUC-ROC", "0.927")
    c3.metric("Gini", "0.854")
    c4.metric("Test set", "420")
    st.caption("Métricas obtenidas sobre el test set de 420 instancias balanceadas con SMOTE.")

    st.markdown("#### Matriz de confusión (test set)")
    cm = pd.DataFrame(
        {
            "": ["Real: Solvente", "Real: Moroso"],
            "Pred: Solvente": [177, 41],
            "Pred: Moroso": [26, 176],
        }
    ).set_index("")
    st.dataframe(cm, use_container_width=True)

    st.markdown(
        """
        #### Preprocesamiento

        El pipeline replica el flujo aplicado en WEKA:

        | Paso | Descripción |
        |---|---|
        | 1. Carga | Dataset German Credit (1000 instancias) |
        | 2. Limpieza | Sin valores faltantes |
        | 3. Encoding | OrdinalEncoder en 13 categóricas |
        | 4. Normalización | MinMaxScaler en 7 numéricas |
        | 5. Balanceo | SMOTE (300 bad → 700 bad sintéticas) |
        | 6. Split | 70% training (980) / 30% test (420) estratificado |
        | 7. Modelo | Random Forest (100 árboles, `class_weight=balanced`) |
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

        **Curso:** Aprendizaje Estadístico
        **Universidad:** UPAO (Universidad Privada Antenor Orrego)
        **Modelo de referencia:** Random Forest entrenado en WEKA sobre German Credit Data

        #### 📊 Sobre el dataset

        El **German Credit Data** es un dataset clásico de clasificación binaria
        para evaluación de credit scoring. Contiene 1000 solicitudes de crédito
        en Alemania con 20 atributos predictivos y 1 variable objetivo:

        - **good** (700 instancias, 70%): cliente solvente
        - **bad** (300 instancias, 30%): cliente moroso

        Los datos incluyen información demográfica, financiera y de historial
        crediticio del solicitante.

        #### 🔄 Notas metodológicas

        1. **SMOTE antes del split:** se replica el flujo de WEKA para
           comparabilidad de métricas. Académicamente lo correcto sería
           aplicar SMOTE solo en training.

        2. **Doble balanceo:** SMOTE + `class_weight='balanced'`. Refuerza
           la atención a la clase minoritaria.

        3. **Test set no representativo del mundo real:** tras SMOTE, el
           test queda 50/50. En producción los datos serían 70/30.

        4. **Validación contra WEKA:** las métricas de Python se comparan
           con las de WEKA (~81.19% accuracy). Variaciones de ±2-3% son
           normales por diferencias de implementación.

        #### 🔗 Fuente del dataset

        - WEKA: [credit-g.arff](https://raw.githubusercontent.com/Waikato/weka-3.8/master/wekadocs/data/credit-g.arff)
        - UCI ML Repository: [Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
        - Documentación: Hofmann, H. (1994)
        """
    )
