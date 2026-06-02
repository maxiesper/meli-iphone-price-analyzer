import pandas as pd
import re

def process_data(file_path):
    print(f"📂 Cargando dataset: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error al leer archivo: {e}")
        return None

    # =========================
    # VALIDACIÓN ESTRUCTURAL
    # =========================
    required_cols = [
        'id', 'title', 'price', 'currency',
        'state_name', 'shipping_free',
        'seller_status', 'condition'
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Columnas faltantes: {missing}")
        return None

    print(f"✅ Dataset cargado: {len(df)} filas")

    # =========================
    # LIMPIEZA INICIAL
    # =========================
    df['title'] = df['title'].astype(str).str.lower()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # eliminar nulos críticos
    df = df.dropna(subset=['title', 'price'])

    # =========================
    # FILTRO: SOLO IPHONE 15
    # =========================
    df = df[df['title'].str.contains(r'iphone\s?15', na=False)].copy()

    # =========================
    # LIMPIEZA SEMÁNTICA
    # =========================
    basura = [
        r'\bfunda\b', r'\bcase\b', r'\bvidrio\b', r'\btemplado\b',
        r'\bcaja\b', r'\bdisplay\b', r'\bmodulo\b', r'\brepuesto\b',
        r'para\s?iphone', r'\bpack\b', r'\bcombo\b'
    ]

    pattern_basura = '|'.join(basura)
    df = df[~df['title'].str.contains(pattern_basura, na=False)].copy()

    # =========================
    # FILTROS DE NEGOCIO
    # =========================
    df = df[df['currency'] == 'ARS'].copy()
    df = df[(df['price'] > 200000) & (df['price'] < 4500000)].copy()

    # =========================
    # FEATURE ENGINEERING
    # =========================
    def extract_storage(title):
        match = re.search(r'\b(64|128|256|512|1024)\s?gb\b', title)
        return int(match.group(1)) if match else None

    def get_model_type(title):
        if re.search(r'\bpro max\b', title):
            return 'Pro Max'
        elif re.search(r'\bpro\b', title):
            return 'Pro'
        elif re.search(r'\bplus\b', title):
            return 'Plus'
        else:
            return 'Base'

    df['storage_gb'] = df['title'].apply(extract_storage)
    df['model_type'] = df['title'].apply(get_model_type)

    # eliminar filas sin features clave
    df = df[df['storage_gb'].notna()].copy()

    # =========================
    # BASELINE INTELIGENTE (POR SEGMENTO)
    # =========================
    df['median_segment'] = df.groupby(
        ['model_type', 'storage_gb']
    )['price'].transform('median')

    df['is_anomaly'] = df['price'] < (df['median_segment'] * 0.7)

    # =========================
    # LOGS DE CONTROL
    # =========================
    print("\n📊 RESUMEN POST-WRANGLING")
    print(f"Filas finales: {len(df)}")

    print("\nDistribución modelos:")
    print(df['model_type'].value_counts())

    print("\nDistribución storage:")
    print(df['storage_gb'].value_counts())

    print("\n% anomalías detectadas:")
    print(round(df['is_anomaly'].mean() * 100, 2), "%")

    return df


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    df_clean = process_data("data_raw_meli.csv")

    if df_clean is not None:
        print("\n--- SAMPLE ---")
        print(df_clean[['title', 'price', 'model_type', 'storage_gb', 'is_anomaly']].head(10))

        df_clean.to_csv("data_raw_meli.csv", index=False)
        print("\n📂 Dataset limpio guardado: data_raw_meli.csv")