import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# =========================
# 📥 CARGA DE DATA
# =========================
def load_data(path):
    print(f"📂 Cargando dataset limpio: {path}")
    df = pd.read_csv(path)
    print(f"✅ Filas: {len(df)}")
    return df


# =========================
# 🧠 FEATURE ENGINEERING
# =========================
def prepare_features(df):
    df = df.copy()

    # Encoding categóricas
    le_model = LabelEncoder()
    le_seller = LabelEncoder()

    df['model_type_enc'] = le_model.fit_transform(df['model_type'])
    df['seller_status'] = df['seller_status'].fillna("unknown")
    df['seller_status_enc'] = le_seller.fit_transform(df['seller_status'])

    # Boolean → int
    df['shipping_free'] = df['shipping_free'].astype(int)

    # Feature potente
    df['price_per_gb'] = df['price'] / df['storage_gb']

    features = [
        'storage_gb',
        'model_type_enc',
        'seller_status_enc',
        'shipping_free',
        'price_per_gb'
    ]

    X = df[features]
    y = df['price']

    return X, y, df


# =========================
# 🌲 MODELO DE REGRESIÓN
# =========================
def train_model(X, y):
    print("\n🌲 Entrenando Random Forest...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"📉 MAE (error medio): {round(mae, 2)}")

    return model


# =========================
# 🚨 DETECCIÓN DE ANOMALÍAS
# =========================
def detect_anomalies(model, X, df):
    print("\n🚨 Detectando anomalías...")

    df = df.copy()

    # Predicción
    df['predicted_price'] = model.predict(X)

    # Error absoluto
    df['error'] = abs(df['price'] - df['predicted_price'])

    # Threshold dinámico (percentil 90)
    threshold = df['error'].quantile(0.90)

    df['is_anomaly_model'] = df['error'] > threshold

    print(f"📊 Threshold error: {round(threshold, 2)}")
    print(f"⚠️ % anomalías (modelo): {round(df['is_anomaly_model'].mean()*100,2)}%")

    return df


# =========================
# 🧬 ISOLATION FOREST
# =========================
def isolation_forest(df, X):
    print("\n🧬 Aplicando Isolation Forest...")

    iso = IsolationForest(
        contamination=0.07,
        random_state=42
    )

    df['anomaly_iso'] = iso.fit_predict(X)

    # -1 = anomalía
    df['is_anomaly_iso'] = df['anomaly_iso'] == -1

    print(f"⚠️ % anomalías (ISO): {round(df['is_anomaly_iso'].mean()*100,2)}%")

    return df


# =========================
# 🎯 MAIN PIPELINE
# =========================
if __name__ == "__main__":

    # 1. Cargar data limpia
    df = load_data("data_clean_meli.csv")

    # 2. Preparar features
    X, y, df = prepare_features(df)

    # 3. Entrenar modelo
    model = train_model(X, y)

    # 4. Detectar anomalías (modelo)
    df = detect_anomalies(model, X, df)

    # 5. Isolation Forest
    df = isolation_forest(df, X)

    # =========================
    # 📊 OUTPUT FINAL
    # =========================
    print("\n--- TOP ANOMALÍAS ---")
    print(
        df.sort_values(by="error", ascending=False)[
            ['title', 'price', 'predicted_price', 'error']
        ].head(10)
    )

    # Guardado final
    df.to_csv("data_model_output.csv", index=False)

    print("\n📂 Dataset final guardado: data_model_output.csv")




