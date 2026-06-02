import pandas as pd

# =========================
# CONFIG
# =========================
FILE_PATH = "data_raw_meli.csv"


# =========================
# 📥 INGESTA LOCAL
# =========================
def load_local_data(file_path):
    print(f"📂 Cargando dataset desde: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("❌ Archivo no encontrado.")
        return None

    print(f"✅ Dataset cargado: {len(df)} filas")

    return df


# =========================
# 🔍 VALIDACIÓN INICIAL
# =========================
def validate_data(df):
    print("\n--- VALIDACIÓN INICIAL ---")

    print("\nColumnas:")
    print(df.columns.tolist())

    print("\nValores nulos:")
    print(df.isnull().sum())

    print("\nMonedas:")
    print(df['currency'].value_counts())

    print("\nEstados:")
    print(df['state_name'].value_counts())

    print("\nShipping free:")
    print(df['shipping_free'].value_counts())

    print("\nSeller status:")
    print(df['seller_status'].value_counts())


# =========================
# 🚀 MAIN
# =========================
if __name__ == "__main__":
    df = load_local_data(FILE_PATH)

    if df is not None:
        validate_data(df)

        print("\n--- SAMPLE ---")
        print(df.head(5))