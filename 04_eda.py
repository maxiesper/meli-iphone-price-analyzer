import pandas as pd
import numpy as np

def run_eda(file_path):

    print("Cargando dataset...")
    df = pd.read_csv(file_path)

    print("\n=== INFO GENERAL ===")
    print(f"Filas: {len(df)}")
    print(f"Columnas: {list(df.columns)}")

    # =========================
    # DISTRIBUCIÓN GENERAL
    # =========================
    print("\n=== DISTRIBUCION DE PRECIOS ===")
    mean_price = df['price'].mean()
    median_price = df['price'].median()
    std_price = df['price'].std()

    print(f"Media: {mean_price:,.0f}")
    print(f"Mediana: {median_price:,.0f}")
    print(f"Desviación estándar: {std_price:,.0f}")

    if mean_price > median_price:
        print("Distribución sesgada a la derecha")
    else:
        print("Distribución simétrica o sesgada a la izquierda")

    # =========================
    # OUTLIERS (IQR)
    # =========================
    print("\n=== OUTLIERS (METODO IQR) ===")

    q1 = df['price'].quantile(0.25)
    q3 = df['price'].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[(df['price'] < lower) | (df['price'] > upper)]

    print(f"Q1: {q1:,.0f}")
    print(f"Q3: {q3:,.0f}")
    print(f"IQR: {iqr:,.0f}")
    print(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

    # =========================
    # ANALISIS POR SEGMENTOS
    # =========================
    print("\n=== MODELO vs PRECIO ===")
    print(df.groupby('model_type')['price'].median().sort_values())

    print("\n=== STORAGE vs PRECIO ===")
    print(df.groupby('storage_gb')['price'].median().sort_values())

    print("\n=== SHIPPING FREE ===")
    print(df.groupby('shipping_free')['price'].median())

    print("\n=== SELLER STATUS ===")
    print(df.groupby('seller_status')['price'].median())

    print("\n=== UBICACION ===")
    print(df.groupby('state_name')['price'].median().sort_values())

    # =========================
    # CORRELACIONES
    # =========================
    print("\n=== CORRELACION ===")
    corr = df[['price', 'storage_gb', 'shipping_free']].corr()
    print(corr)

    # =========================
    # INTERACCIONES IMPORTANTES
    # =========================
    print("\n=== INTERACCION MODELO + STORAGE ===")

    combo = df.groupby(['model_type', 'storage_gb'])['price'].median().unstack()
    print(combo)

    # =========================
    # ANOMALIAS
    # =========================
    print("\n=== ANALISIS DE ANOMALIAS ===")

    anomaly_rate = df['is_anomaly'].mean() * 100
    print(f"Tasa de anomalías: {anomaly_rate:.2f}%")

    print("\nPrecio medio NORMAL vs ANOMALO:")
    print(df.groupby('is_anomaly')['price'].mean())

    # =========================
    # INSIGHT AUTOMATICO
    # =========================
    print("\n=== INSIGHTS AUTOMATICOS ===")

    strongest_corr = corr['price'].drop('price').abs().idxmax()
    print(f"Variable más influyente en precio: {strongest_corr}")

    if anomaly_rate < 5:
        print("Sistema conservador (detecta pocas anomalías)")
    elif anomaly_rate < 15:
        print("Sistema balanceado")
    else:
        print("Sistema agresivo (muchas anomalías)")

    print("\nEDA finalizado.")


if __name__ == "__main__":
    run_eda("data_clean_meli.csv")