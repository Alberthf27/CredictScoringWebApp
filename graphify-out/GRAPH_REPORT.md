# Graph Report - .  (2026-06-23)

## Corpus Check
- Corpus is ~1,494 words - fits in a single context window. You may not need a graph.

## Summary
- 25 nodes · 26 edges · 3 communities
- Extraction: 65% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Dependencias del proyecto (requirements.txt + paquetes)|Dependencias del proyecto (requirements.txt + paquetes)]]
- [[_COMMUNITY_Núcleo de la app Streamlit (UI, prediccion, modelos)|Núcleo de la app Streamlit (UI, prediccion, modelos)]]
- [[_COMMUNITY_Carga y validación de datos (CSVARFF, preprocessor)|Carga y validación de datos (CSV/ARFF, preprocessor)]]

## God Nodes (most connected - your core abstractions)
1. `cargar_csv()` - 3 edges
2. `cargar_arff()` - 3 edges
3. `validar_columnas()` - 3 edges
4. `predecir_lote()` - 3 edges
5. `load_models()` - 2 edges
6. `colorear_filas()` - 2 edges
7. `Credit Scoring - German Credit Data Aplicación Streamlit para predicción por lot` - 1 edges
8. `Carga los 4 archivos .pkl exportados desde el notebook.` - 1 edges
9. `Carga un CSV desde un archivo subido a Streamlit.` - 1 edges
10. `Carga un ARFF desde un archivo subido a Streamlit.` - 1 edges

## Surprising Connections (you probably didn't know these)
- `predecir_lote()` --references--> `DataFrame`  [EXTRACTED]
  app.py →   _Bridges community 2 → community 1_

## Import Cycles
- None detected.

## Communities (3 total, 0 thin omitted)

### Community 0 - "Dependencias del proyecto (requirements.txt + paquetes)"
Cohesion: 0.20
Nodes (10): imbalanced-learn, joblib, matplotlib, numpy, pandas, scikit-learn, scipy, seaborn (+2 more)

### Community 1 - "Núcleo de la app Streamlit (UI, prediccion, modelos)"
Cohesion: 0.25
Nodes (7): colorear_filas(), load_models(), predecir_lote(), Credit Scoring - German Credit Data Aplicación Streamlit para predicción por lot, Transforma y predice para todo el DataFrame. Devuelve un DF con resultados., Devuelve colores para una fila según la predicción., Carga los 4 archivos .pkl exportados desde el notebook.

### Community 2 - "Carga y validación de datos (CSV/ARFF, preprocessor)"
Cohesion: 0.29
Nodes (7): cargar_arff(), cargar_csv(), Carga un CSV desde un archivo subido a Streamlit., Carga un ARFF desde un archivo subido a Streamlit., Retorna la lista de columnas requeridas que faltan en el DataFrame., validar_columnas(), DataFrame

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `cargar_csv()` connect `Carga y validación de datos (CSV/ARFF, preprocessor)` to `Núcleo de la app Streamlit (UI, prediccion, modelos)`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `cargar_arff()` connect `Carga y validación de datos (CSV/ARFF, preprocessor)` to `Núcleo de la app Streamlit (UI, prediccion, modelos)`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `validar_columnas()` connect `Carga y validación de datos (CSV/ARFF, preprocessor)` to `Núcleo de la app Streamlit (UI, prediccion, modelos)`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **What connects `Credit Scoring - German Credit Data Aplicación Streamlit para predicción por lot`, `Carga los 4 archivos .pkl exportados desde el notebook.`, `Carga un CSV desde un archivo subido a Streamlit.` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._