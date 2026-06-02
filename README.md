# 📱 MeLi iPhone Price Analyzer

> Sistema de inteligencia de precios para el ecosistema MercadoLibre Argentina.  
> Detecta anomalías, predice precios esperados y emite recomendaciones accionables según el rol del usuario.

---

## 🧠 ¿Qué hace este proyecto?

Toma una publicación de iPhone 15 en MercadoLibre y responde tres preguntas:

- ¿Este precio es correcto para el mercado?
- ¿Es una oportunidad real o una anomalía sospechosa?
- ¿Qué debería hacer con él?

La respuesta cambia según quién pregunte: un **vendedor**, un **comprador**, o un **analista del marketplace**.

---

## 🎯 ¿Para quién es?

| Rol | Uso |
|-----|-----|
| **Vendedor** | Optimizar precio para ganar competitividad sin destruir margen |
| **Comprador** | Detectar ofertas reales vs publicaciones fraudulentas |
| **Analista / MeLi** | Identificar publicaciones anómalas para revisión manual |

---

## 🏗️ Arquitectura del sistema

```
dataset generado simulacion mercado libre.py
       ↓
data_raw_meli.csv
       ↓
02_wrangling.py       → limpieza y feature engineering
       ↓
data_clean_meli.csv
       ↓
05_dashboard.py       → modelo + validación + interfaz
       ↓
Dashboard Streamlit   → evaluación · mercado · anomalías · validación
```

---

## ⚙️ Stack tecnológico

| Categoría | Herramienta |
|-----------|-------------|
| Lenguaje | Python 3.12 |
| Dashboard | Streamlit |
| ML | scikit-learn (GradientBoosting, IsolationForest) |
| Visualización | Plotly Express / Plotly Graph Objects |
| Datos | Pandas, NumPy |
| Validación | KFold Cross Validation (5 folds) |

---

## 🤖 Modelos utilizados

### GradientBoostingRegressor — predicción de precio esperado
Aprende el precio "justo" de cada publicación en función de:
- Modelo del iPhone (Base / Plus / Pro / Pro Max)
- Capacidad de almacenamiento (128 / 256 / 512 GB)
- Reputación del vendedor (platinum / gold / silver)
- Envío gratis (sí / no)

### IsolationForest — detección de anomalías
Detecta publicaciones que no encajan con el patrón general del mercado.
Clasifica cada anomalía con un motivo probable:
- Precio bajo + sin reputación
- Precio muy bajo de vendedor platinum (sospechoso)
- Precio excesivamente alto
- Sin envío + precio elevado

---

## 📊 Métricas del modelo

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| R² (CV-5) | 0.856 | El modelo explica el 85.6% de la variabilidad de precios |
| MAE | $66.000 ARS | Error promedio por publicación |
| Error relativo | ~5% | En un iPhone de $1.3M, el modelo se equivoca ~$66K |

> Validado con 5-fold cross validation. Los folds muestran resultados consistentes, sin señales de overfitting.

---

## 🖥️ Dashboard — 4 tabs

### Tab 1 · Evaluador de Precio
Ingresás modelo, storage, precio, vendedor y envío.
El sistema devuelve:
- **Veredicto:** MANTENER / AJUSTAR / REVISAR
- **Recomendación accionable** según tu rol
- **Gauge** del índice de oportunidad (0-100 pts)
- Precio esperado por ML, percentil en el segmento, score de Isolation Forest
- Posición respecto al rango IQR del segmento

### Tab 2 · Vista de Mercado
- Scatter: precio vs índice de oportunidad por modelo
- Boxplot: distribución de precios por provincia
- Tabla completa con gradiente de oportunidad

### Tab 3 · Análisis de Anomalías
- KPIs del universo anómalo
- Filtros por modelo y motivo
- Gráfico de barras por tipo de anomalía
- Tabla detallada con colores por nivel de desviación

### Tab 4 · Validación del Modelo
- R², MAE y RMSE en lenguaje de negocio
- Gráficos de R² y MAE por fold (evidencia de ausencia de overfitting)
- Distribución real de precios por modelo
- Nota metodológica completa

---

## 🚀 Cómo correr el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/maximiliano-esper/meli-iphone-price-analyzer.git
cd meli-iphone-price-analyzer
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Generar el dataset
```bash
python "dataset generado simulacion mercado libre.py"
```

### 4. Correr el wrangling
```bash
python 02_wrangling.py
```

### 5. Lanzar el dashboard
```bash
streamlit run 05_dashboard.py
```

---

## 📁 Estructura del proyecto

```
meli-iphone-price-analyzer/
│
├── dataset generado simulacion mercado libre.py   # generador de datos sintéticos
├── 01_extraction.py                               # extracción de datos
├── 02_wrangling.py                                # limpieza y feature engineering
├── 03_modeling.py                                 # exploración de modelos
├── 04_eda.py                                      # análisis exploratorio
├── 05_dashboard.py                                # dashboard principal (Streamlit)
│
├── data_raw_meli.csv                              # dataset crudo generado
├── data_clean_meli.csv                            # dataset limpio (generado por wrangling)
│
├── graficos/                                      # gráficos exportados del EDA
│
├── requirements.txt                               # dependencias del proyecto
└── README.md                                      # este archivo
```

---

## 📌 Decisiones técnicas destacadas

**Encoding consistente entre train y predict**
Se usa `pd.Categorical` con categorías fijas (`ALL_MODELS`) para garantizar que las columnas dummy sean idénticas en entrenamiento y predicción, evitando errores de alineación.

**`st.cache_resource` vs `st.cache_data`**
Los pipelines de sklearn se cachean con `cache_resource` (objetos stateful). El DataFrame se cachean con `cache_data` (serializable). La distinción evita reentrenamientos innecesarios en cada interacción.

**Mediana de referencia sobre dataset completo**
El índice de oportunidad se calcula contra la mediana del dataset completo, no del subset filtrado. Esto garantiza comparaciones justas independientemente de los filtros activos.

**Lógica de veredicto en 3 señales**
El veredicto combina: desviación vs modelo ML, posición IQR estadístico, y flag de Isolation Forest. Cada señal aporta una dimensión distinta del análisis.

---

## 👤 Autor

**Maximiliano Esper**  
Tecnicatura en Ciencia de Datos · Universidad Nacional de Salta  
Licenciado en Ciencias de la Comunicación · UNSA  

[LinkedIn](https://linkedin.com/in/maximiliano-esper) · [GitHub](https://github.com/maximiliano-esper)

---

## 📄 Licencia

MIT — libre para usar, modificar y distribuir con atribución.
