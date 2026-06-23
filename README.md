# 🏦 Credit Scoring Web App

Aplicación web de **Credit Scoring** construida con Streamlit que replica un modelo de Random Forest previamente entrenado en WEKA sobre el dataset **German Credit Data**.

## 🎯 Descripción

Este proyecto implementa un sistema de scoring crediticio que:

- Acepta archivos **CSV** o **ARFF** con datos de clientes
- Procesa los datos a través de un pipeline de ML entrenado
- Devuelve la predicción (`SOLVENTE` / `MOROSO`) con probabilidades para cada cliente
- Permite descargar los resultados como CSV

## 📊 Métricas del modelo

| Métrica | Valor |
|---|---|
| **Accuracy** | 84.29% |
| **AUC-ROC** | 0.9251 |
| **Gini** | 0.8501 |
| **Algoritmo** | Random Forest (100 árboles) |
| **Training set** | 980 instancias (post-SMOTE) |
| **Test set** | 420 instancias (post-SMOTE) |

## 🏗️ Arquitectura

```
┌──────────────────┐
│  Usuario / API   │
└────────┬─────────┘
         │ Sube CSV/ARFF
         ▼
┌──────────────────┐
│  Streamlit App   │  ← app.py
│  (cargar +       │
│   validar + UI)  │
└────────┬─────────┘
         │ preprocessor.transform()
         ▼
┌──────────────────┐
│  preprocessor.pkl│  ← MinMaxScaler + OrdinalEncoder
└────────┬─────────┘
         │ model.predict()
         ▼
┌──────────────────┐
│ random_forest_   │  ← RandomForestClassifier
│ model.pkl        │     (100 trees, balanced)
└────────┬─────────┘
         │ Predicción + probabilidades
         ▼
┌──────────────────┐
│  Tabla de        │  ← Descargable como CSV
│  resultados      │
└──────────────────┘
```

## 📁 Estructura del proyecto

```
.
├── app.py                          # Aplicación Streamlit principal
├── modelo_creditscoring.ipynb      # Notebook de entrenamiento
├── requirements.txt                # Dependencias de Python
├── README.md                       # Este archivo
├── .gitignore                      # Exclusiones de Git
│
├── preprocessor.pkl                # Pipeline de preprocesamiento fiteado
├── random_forest_model.pkl         # Modelo Random Forest entrenado
├── feature_names.pkl               # Nombres de features
├── categorical_options.pkl         # Opciones para los selectbox
│
├── clientes_ejemplo.csv            # CSV de prueba (10 clientes)
│
└── graphify-out/                   # Grafo de conocimiento del proyecto
    ├── graph.html                  # Visualización interactiva
    ├── graph.json                  # Datos crudos del grafo
    └── GRAPH_REPORT.md             # Reporte del grafo
```

## 🚀 Uso local

### 1. Clonar e instalar dependencias

```bash
git clone https://github.com/Alberthf27/CredictScoringWebApp.git
cd CredictScoringWebApp
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`.

### 3. Probar con datos de ejemplo

Sube el archivo `clientes_ejemplo.csv` (10 clientes de muestra) en la interfaz web y haz clic en **"Analizar Riesgo Crediticio"**.

## 📋 Formato del archivo de entrada

El archivo debe contener las **20 columnas requeridas** por el modelo. Columnas adicionales (como `cliente_id` o `nombre`) se preservan en la salida pero no se usan para la predicción.

### Variables categóricas (13)

| Variable | Valores posibles |
|---|---|
| `checking_status` | `<0`, `0<=X<200`, `>=200`, `no checking` |
| `credit_history` | `no credits/all paid`, `all paid`, `existing paid`, `delayed previously`, `critical/other existing credit` |
| `purpose` | `new car`, `used car`, `furniture/equipment`, `radio/tv`, `domestic appliance`, `repairs`, `education`, `vacation`, `retraining`, `business`, `other` |
| `savings_status` | `<100`, `100<=X<500`, `500<=X<1000`, `>=1000`, `no known savings` |
| `employment` | `unemployed`, `<1`, `1<=X<4`, `4<=X<7`, `>=7` |
| `personal_status` | `male div/sep`, `female div/dep/mar`, `male single`, `male mar/wid`, `female single` |
| `other_parties` | `none`, `co applicant`, `guarantor` |
| `property_magnitude` | `real estate`, `life insurance`, `car`, `no known property` |
| `other_payment_plans` | `bank`, `stores`, `none` |
| `housing` | `rent`, `own`, `for free` |
| `job` | `unemp/unskilled non res`, `unskilled resident`, `skilled employee/official`, `high qualif/self emp/mgmt` |
| `own_telephone` | `none`, `yes` |
| `foreign_worker` | `yes`, `no` |

### Variables numéricas (7)

| Variable | Rango típico |
|---|---|
| `duration` | 1 – 72 meses |
| `credit_amount` | 250 – 18424 DM |
| `installment_commitment` | 1 – 4 |
| `residence_since` | 1 – 4 años |
| `age` | 18 – 75 años |
| `existing_credits` | 1 – 4 |
| `num_dependents` | 1 – 2 |

## ☁️ Deploy en Streamlit Cloud

Esta aplicación está desplegada en [Streamlit Cloud](https://share.streamlit.io) y se puede acceder desde cualquier navegador sin instalación.

**Pasos para redesplegar:**

1. Conecta este repositorio a [share.streamlit.io](https://share.streamlit.io)
2. Selecciona la rama `main` y el archivo principal `app.py`
3. Streamlit Cloud instalará automáticamente las dependencias de `requirements.txt`
4. La app estará disponible en una URL pública

## 🔄 Reentrenar el modelo

Para reentrenar el modelo desde cero:

1. Abre `modelo_creditscoring.ipynb` en Google Colab o Jupyter
2. Ejecuta todas las celdas en orden
3. Los nuevos archivos `.pkl` se generan al final
4. Reemplaza los archivos `.pkl` en este repositorio y haz push

## 🛠️ Stack tecnológico

- **Python 3.10+**
- **Streamlit 1.28+** - Framework web
- **scikit-learn 1.3+** - Modelo de ML
- **imbalanced-learn 0.12+** - SMOTE
- **pandas / numpy** - Manipulación de datos
- **scipy** - Lectura de ARFF
- **matplotlib / seaborn** - Visualización
- **joblib** - Persistencia del modelo

## 📚 Dataset

**German Credit Data** (1000 instancias, 20 features + 1 target)

- Fuente: [Weka - credit-g.arff](https://raw.githubusercontent.com/Waikato/weka-3.8/master/wekadocs/data/credit-g.arff)
- Documentación: Hofmann, H. (1994). *Statlog (German Credit Data)*
- Distribución original: 700 `good` (70%) / 300 `bad` (30%)
- Tras SMOTE: 700 `good` (50%) / 700 `bad` (50%)

## 📝 Notas metodológicas

1. **SMOTE aplicado antes del train/test split** — Replica el flujo de WEKA. Lo académicamente correcto sería aplicar SMOTE solo en training para evitar data leakage.
2. **Doble balanceo** — SMOTE + `class_weight='balanced'`. Refuerza la atención a la clase minoritaria.
3. **Test set no representativo del mundo real** — Tras SMOTE, el test queda 50/50. Las métricas son optimistas vs. producción.
4. **Validación contra WEKA** — Las métricas de Python se comparan con las de WEKA (~81.19% accuracy, ~0.895 AUC, ~0.790 Gini). Variaciones de ±2-3% son normales.

## 👤 Autor

- **Curso:** Aprendizaje Estadístico
- **Universidad:** UPAO (Universidad Privada Antenor Orrego)
- **Modelo de referencia:** Random Forest entrenado en WEKA

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub.
