import pandas as pd
import numpy as np
import random

# =========================
# CONFIG
# =========================
random.seed(42)
np.random.seed(42)

N = 3000

models        = ["Base", "Plus", "Pro", "Pro Max"]
storages      = [128, 256, 512]
states        = ["CABA", "Buenos Aires", "Córdoba", "Santa Fe", "Mendoza", "Salta"]
seller_status = ["platinum", "gold", "silver", None]

# Precios base realistas por combinación
base_prices = {
    ("Base",    128): 900_000,
    ("Base",    256): 1_050_000,
    ("Plus",    128): 1_100_000,
    ("Plus",    256): 1_250_000,
    ("Pro",     128): 1_300_000,
    ("Pro",     256): 1_500_000,
    ("Pro",     512): 1_800_000,
    ("Pro Max", 256): 1_700_000,
    ("Pro Max", 512): 2_100_000,
}

# Combinaciones válidas (no todos los modelos tienen todos los storages)
valid_combos = list(base_prices.keys())

def generate_title(model, storage):
    noise = ["", "nuevo sellado", "libre", "garantia oficial",
             "caja abierta", "oferta", "envio gratis", "full"]
    return f"iphone 15 {model.lower()} {storage}gb {random.choice(noise)}".strip()

def generate_price(base):
    """
    Ruido reducido (±5% normal) con anomalías poco frecuentes (2%).
    Esto da precios realistas y un modelo con R² > 0.75.
    """
    variation = random.uniform(0.95, 1.05)   # ±5% variación normal
    price = base * variation

    # Anomalías reales pero poco frecuentes
    anomaly = random.random()
    if anomaly < 0.02:                              # 2% muy baratos (fraude/dañado)
        price *= random.uniform(0.5, 0.7)
    elif anomaly > 0.98:                            # 2% muy caros (error/mal categorizados)
        price *= random.uniform(1.4, 1.8)

    return int(price)

def generate_seller_by_model(model):
    """
    Vendedores platinum venden más Pro y Pro Max (más realista).
    """
    if model in ["Pro", "Pro Max"]:
        return random.choices(
            ["platinum", "gold", "silver", None],
            weights=[0.45, 0.30, 0.15, 0.10]
        )[0]
    else:
        return random.choices(
            ["platinum", "gold", "silver", None],
            weights=[0.25, 0.35, 0.25, 0.15]
        )[0]

def generate_shipping_by_seller(seller):
    """Platinum casi siempre tiene envío gratis."""
    if seller == "platinum":
        return random.random() < 0.90
    elif seller == "gold":
        return random.random() < 0.65
    elif seller == "silver":
        return random.random() < 0.40
    else:
        return random.random() < 0.20

data = []

for _ in range(N):
    model, storage = random.choice(valid_combos)
    base_price     = base_prices[(model, storage)]
    seller         = generate_seller_by_model(model)
    shipping       = generate_shipping_by_seller(seller)

    data.append({
        "id":            f"MLA{random.randint(1000000000, 9999999999)}",
        "title":         generate_title(model, storage),
        "model_type":    model,
        "storage_gb":    storage,
        "price":         generate_price(base_price),
        "currency":      "ARS",
        "state_name":    random.choice(states),
        "shipping_free": shipping,
        "logistic_type": "fulfillment" if shipping else "self_service",
        "seller_status": seller,
        "condition":     "new",
        "permalink":     "https://articulo.mercadolibre.com.ar/fake"
    })

df = pd.DataFrame(data)
df.to_csv("data_clean_meli.csv", index=False, encoding="utf-8-sig")

print(f"✅ Dataset regenerado con {len(df)} filas")
print(f"\nDistribución de modelos:\n{df['model_type'].value_counts()}")
print(f"\nRango de precios por modelo:")
print(df.groupby("model_type")["price"].agg(["min", "median", "max"]).applymap(lambda x: f"${x:,.0f}"))
print(f"\nVendedores:\n{df['seller_status'].value_counts(dropna=False)}")