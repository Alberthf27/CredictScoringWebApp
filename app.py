"""
Credit Scoring - German Credit Data
Aplicación Streamlit para predicción por lotes (batch)
Acepta archivos CSV y ARFF, devuelve tabla de resultados con descarga.
"""

import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import arff


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Credit Scoring - German Credit",
    page_icon="\U0001F3E6",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ESTILOS CSS PERSONALIZADOS
# ============================================================
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e6da4 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white !important; margin: 0; }
    .main-header p  { color: #d6e4f0 !important; margin: 0.3rem 0 0 0; }
    .metric-card {
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
    }
    .stButton>button {
        background-color: #1f4e79;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
    }
    .stButton>button:hover { background-color: #2e6da4; }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CARGA DE MODELOS (con caché y manejo de errores)
# ============================================================
@st.cache_resource(show_spinner="Cargando modelo entrenado...")
def load_models():
    """Carga los 4 archivos .pkl exportados desde el notebook."""
    try:
        preprocessor = joblib.load("preprocessor.pkl")
        model = joblib.load("random_forest_model.pkl")
        feature_names = joblib.load("feature_names.pkl")
        categorical_options = joblib.load("categorical_options.pkl")
        return preprocessor, model, feature_names, categorical_options
    except FileNotFoundError as e:
        st.error(
            f"\u274C No se encontró el archivo: **{e.filename}**\n\n"
            "Asegúrate de colocar los 4 archivos `.pkl` en la misma carpeta que `app.py`:\n"
            "- `preprocessor.pkl`\n"
            "- `random_forest_model.pkl`\n"
            "- `feature_names.pkl`\n"
            "- `categorical_options.pkl`"
        )
        st.stop()
    except Exception as e:
        st.error(f"\u274C Error al cargar el modelo: {e}")
        st.stop()


preprocessor, model, feature_names, categorical_options = load_models()


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def cargar_csv(archivo) -> pd.DataFrame:
    """Carga un CSV desde un archivo subido a Streamlit."""
    try:
        return pd.read_csv(archivo)
    except UnicodeDecodeError:
        return pd.read_csv(archivo, encoding="latin-1")


def cargar_arff(archivo) -> pd.DataFrame:
    """Carga un ARFF desde un archivo subido a Streamlit."""
    contenido = archivo.read().decode("utf-8")
    datos, _meta = arff.loadarff(io.StringIO(contenido))
    df = pd.DataFrame(datos)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode("utf-8")
    return df


def validar_columnas(df: pd.DataFrame) -> list:
    """Retorna la lista de columnas requeridas que faltan en el DataFrame."""
    return [c for c in feature_names if c not in df.columns]


def predecir_lote(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma y predice para todo el DataFrame. Devuelve un DF con resultados."""
    X = df[feature_names].copy()
    X_transformed = preprocessor.transform(X)
    preds = model.predict(X_transformed)
    probas = model.predict_proba(X_transformed)

    clases = model.classes_
    idx_good = list(clases).index("good")
    idx_bad = list(clases).index("bad")

    resultados = df.copy()
    resultados["Prediccion"] = ["SOLVENTE" if p == "good" else "MOROSO" for p in preds]
    resultados["Prob_Solvente_%"] = (probas[:, idx_good] * 100).round(2)
    resultados["Prob_Moroso_%"] = (probas[:, idx_bad] * 100).round(2)
    return resultados


def colorear_filas(row):
    """Devuelve colores para una fila según la predicción."""
    if row["Prediccion"] == "SOLVENTE":
        return ["background-color: #d4edda; color: #155724"] * len(row)
    return ["background-color: #f8d7da; color: #721c24"] * len(row)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## \U0001F3E6 Credit Scoring")
    st.markdown("---")
    st.markdown("### \U0001F4CA Modelo")
    st.metric("Accuracy", "81.19%")
    st.metric("AUC-ROC", "0.895")
    st.metric("Gini", "0.790")
    st.markdown("---")
    st.markdown("### \U0001F4D6 Dataset")
    st.write("German Credit Data")
    st.write("1000 instancias originales")
    st.write("20 features + 1 target")
    st.markdown("---")
    st.markdown("### \U0001F393 Proyecto")
    st.write("**Curso:** Aprendizaje Estadístico")
    st.write("**Universidad:** UPAO")
    st.write("**Algoritmo:** Random Forest")


# ============================================================
# HEADER PRINCIPAL
# ============================================================
st.markdown(
    """
    <div class="main-header">
        <h1>\U0001F3E6 Credit Scoring - German Credit Data</h1>
        <p>Sube un archivo CSV o ARFF con datos de clientes y obtén la predicción de riesgo crediticio</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INSTRUCCIONES
# ============================================================
with st.expander("\U0001F4D6 ¿Cómo funciona? - Click aquí para ver las instrucciones", expanded=False):
    st.markdown(
        """
        ### Flujo de uso

        1. **Prepara tu archivo** con las 20 columnas requeridas (ver pestaña *Descripción de Variables*).
           Puedes subir un archivo **CSV** o **ARFF**.
        2. **Sube el archivo** usando el botón de abajo.
        3. **Verifica** que la vista previa se vea correcta.
        4. **Haz clic** en *"Analizar Riesgo Crediticio"*.
        5. **Descarga** los resultados como CSV.

        ### ¿Qué hace el modelo?

        El modelo **Random Forest** evalúa a cada cliente y devuelve:
        - **Predicción:** `SOLVENTE` (buen pagador) o `MOROSO` (alto riesgo)
        - **Probabilidades:** porcentaje de confianza para cada clase

        El modelo fue entrenado con el dataset German Credit y replicado desde WEKA
        con pipeline de SMOTE + MinMaxScaler + OrdinalEncoder.
        """
    )


# ============================================================
# UPLOAD DE ARCHIVO
# ============================================================
st.markdown("### \U0001F4C2 Subir archivo de datos")
archivo_subido = st.file_uploader(
    "Selecciona un archivo CSV o ARFF con los datos de los clientes",
    type=["csv", "arff"],
    help="El archivo debe contener las 20 columnas requeridas por el modelo"
)

if archivo_subido is not None:
    nombre = archivo_subido.name.lower()

    # Cargar según extensión
    try:
        if nombre.endswith(".arff"):
            df_input = cargar_arff(archivo_subido)
        else:
            df_input = cargar_csv(archivo_subido)
    except Exception as e:
        st.error(f"\u274C No se pudo leer el archivo: {e}")
        st.stop()

    # Quitar columna target si viene incluida
    if "class" in df_input.columns:
        st.warning("\u26A0\uFE0F El archivo contiene la columna `class`. Será ignorada para la predicción.")
        df_input = df_input.drop(columns=["class"])

    # Validar columnas
    faltantes = validar_columnas(df_input)
    if faltantes:
        st.error(
            f"\u274C El archivo no tiene todas las columnas requeridas.\n\n"
            f"**Faltan {len(faltantes)} columnas:**\n" +
            "\n".join(f"- `{c}`" for c in faltantes)
        )
        st.info(
            "**Columnas requeridas (20):**\n" +
            ", ".join(f"`{c}`" for c in feature_names)
        )
    else:
        # Vista previa
        st.success(f"\u2705 Archivo cargado correctamente: **{len(df_input)} clientes** detectados")
        st.markdown("#### \U0001F50D Vista previa de los datos")
        st.dataframe(df_input.head(10), use_container_width=True)

        # Botón de predicción
        st.markdown("---")
        if st.button("\U0001F50D Analizar Riesgo Crediticio", type="primary", use_container_width=True):
            with st.spinner("Procesando predicciones..."):
                try:
                    df_resultados = predecir_lote(df_input)

                    # Métricas del batch
                    n_total = len(df_resultados)
                    n_solv = (df_resultados["Prediccion"] == "SOLVENTE").sum()
                    n_mor = (df_resultados["Prediccion"] == "MOROSO").sum()
                    pct_solv = n_solv / n_total * 100
                    pct_mor = n_mor / n_total * 100

                    st.markdown("### \U0001F4CA Resumen del análisis")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total clientes", n_total)
                    col2.metric("Solventes", f"{n_solv}", f"{pct_solv:.1f}%")
                    col3.metric("Morosos", f"{n_mor}", f"{pct_mor:.1f}%")
                    col4.metric("Riesgo promedio", f"{df_resultados['Prob_Moroso_%'].mean():.1f}%")

                    # Tabla de resultados con colores
                    st.markdown("### \U0001F4CB Resultados detallados")
                    styled = df_resultados.style.apply(colorear_filas, axis=1)
                    st.dataframe(styled, use_container_width=True, height=400)

                    # Botón de descarga
                    csv_resultados = df_resultados.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="\U0001F4E5 Descargar resultados como CSV",
                        data=csv_resultados,
                        file_name="resultados_credit_scoring.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )

                    # Distribución de riesgos
                    st.markdown("### \U0001F4C8 Distribución de probabilidad de morosidad")
                    bins = [0, 20, 40, 60, 80, 100]
                    labels_bins = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
                    df_resultados["Rango_riesgo"] = pd.cut(
                        df_resultados["Prob_Moroso_%"], bins=bins, labels=labels_bins
                    )
                    distribucion = df_resultados["Rango_riesgo"].value_counts().sort_index()
                    st.bar_chart(distribucion)

                except Exception as e:
                    st.error(f"\u274C Error al procesar las predicciones: {e}")
                    st.exception(e)


# ============================================================
# PESTAÑAS INFORMATIVAS
# ============================================================
st.markdown("---")
st.markdown("## \U0001F4DA Información del Proyecto")
tab1, tab2, tab3 = st.tabs(
    ["\U0001F4CA Sobre el Modelo", "\U0001F4CB Descripción de Variables", "\U0001F4C4 Acerca del Proyecto"]
)

with tab1:
    st.markdown("### Métricas del modelo Random Forest")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "81.19%", help="Porcentaje de predicciones correctas")
    col2.metric("AUC-ROC", "0.895", help="Área bajo la curva ROC")
    col3.metric("Gini", "0.790", help="2*AUC - 1, poder discriminante")
    col4.metric("Test size", "420", help="Instancias de test (30%)")

    st.markdown("### Matriz de confusión (test set)")
    cm_data = pd.DataFrame(
        {
            "": ["Real: good", "Real: bad"],
            "Pred: good": [173, 47],
            "Pred: bad": [32, 168]
        }
    ).set_index("")
    st.dataframe(cm_data, use_container_width=True)

    st.markdown(
        """
        ### Hiperparámetros del modelo
        - **n_estimators:** 100 árboles
        - **max_depth:** None (sin límite)
        - **min_samples_leaf:** 1
        - **class_weight:** balanced
        - **random_state:** 42

        ### Preprocesamiento
        - **MinMaxScaler** en variables numéricas
        - **OrdinalEncoder** en variables categóricas
        - **SMOTE** para balanceo de clases (300 bad → 700 bad sintéticas)
        - **Split** 70/30 estratificado
        """
    )

with tab2:
    st.markdown("### Las 20 variables del modelo")
    descripciones = {
        "checking_status": ("Categórica", "Estado de cuenta corriente"),
        "duration": ("Numérica", "Duración del crédito en meses"),
        "credit_history": ("Categórica", "Historial crediticio"),
        "purpose": ("Categórica", "Propósito del crédito"),
        "credit_amount": ("Numérica", "Monto del crédito en DM"),
        "savings_status": ("Categórica", "Estado de ahorros"),
        "employment": ("Categórica", "Antigüedad laboral"),
        "installment_commitment": ("Numérica", "Tasa de cuota sobre ingreso disponible"),
        "personal_status": ("Categórica", "Estado personal y sexo"),
        "other_parties": ("Categórica", "Otros deudores / garantes"),
        "residence_since": ("Numérica", "Años en residencia actual"),
        "property_magnitude": ("Categórica", "Magnitud de propiedades"),
        "age": ("Numérica", "Edad en años"),
        "other_payment_plans": ("Categórica", "Otros planes de pago"),
        "housing": ("Categórica", "Tipo de vivienda"),
        "existing_credits": ("Numérica", "Créditos existentes en este banco"),
        "job": ("Categórica", "Tipo de empleo"),
        "num_dependents": ("Numérica", "Número de personas a cargo"),
        "own_telephone": ("Categórica", "Teléfono propio"),
        "foreign_worker": ("Categórica", "Trabajador extranjero")
    }
    filas = []
    for col in feature_names:
        tipo, desc = descripciones.get(col, ("—", "—"))
        filas.append({"Variable": col, "Tipo": tipo, "Descripción": desc})
    df_desc = pd.DataFrame(filas)
    st.dataframe(df_desc, use_container_width=True, hide_index=True)

    st.markdown("### Valores posibles de las variables categóricas")
    for col, opciones in categorical_options.items():
        with st.expander(f"`{col}`"):
            st.write(", ".join(f"`{v}`" for v in opciones))

with tab3:
    st.markdown(
        """
        ### \U0001F393 Acerca del Proyecto

        **Curso:** Aprendizaje Estadístico
        **Universidad:** UPAO (Universidad Privada Antenor Orrego)

        ### \U0001F4CA Sobre el dataset

        El **German Credit Data** es un dataset clásico de clasificación binaria
        utilizado para evaluar algoritmos de credit scoring. Contiene 1000 instancias
        de solicitudes de crédito en Alemania, con 20 atributos predictivos que incluyen:

        - Datos demográficos (edad, estado civil, sexo)
        - Historia crediticia
        - Capacidad financiera (monto, ahorros, empleo)
        - Propósito del crédito

        **Variable objetivo:**
        - `good` (700 instancias, 70%): cliente solvente
        - `bad` (300 instancias, 30%): cliente moroso

        ### \U0001F9EA Metodología

        El pipeline replica exactamente el flujo aplicado en WEKA:
        1. Limpieza de datos
        2. Eliminación de variables con varianza casi nula
        3. **SMOTE** para balanceo de clases
        4. **MinMaxScaler** para normalización
        5. Entrenamiento con **Random Forest** (100 árboles)
        6. Validación con split 70/30 estratificado

        ### \U0001F4D6 Fuente

        - Dataset: [Weka - credit-g.arff](https://raw.githubusercontent.com/Waikato/weka-3.8/master/wekadocs/data/credit-g.arff)
        - Documentación: Hofmann, H. (1994). *Statlog (German Credit Data)*
        """
    )
