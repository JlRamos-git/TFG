import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURACIÓN ---
ruta_v1 = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted.xlsx"
ruta_v2 = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted_v2.xlsx"
output_folder = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\Graficas_Exploratorias"
os.makedirs(output_folder, exist_ok=True)

# --- CARGA DE DATOS ---
v1 = pd.read_excel(ruta_v1)
v2 = pd.read_excel(ruta_v2)

# --- FORMATO DE FECHAS ---
v1["month"] = pd.to_datetime(v1["month"])
v2["month"] = pd.to_datetime(v2["month"])

# --- CALCULAR MESES AJUSTADOS ---
p10 = v1["comment_count"].quantile(0.10)
p90 = v1["comment_count"].quantile(0.90)

ajustes_p10 = v1[v1["comment_count"] <= p10]["month"]
ajustes_p90 = v1[v1["comment_count"] >= p90]["month"]
meses_ajustados = pd.concat([ajustes_p10, ajustes_p90]).drop_duplicates().sort_values()

# --- UNIÓN DE SERIES ---
merged = pd.merge(
    v1[["month", "sentiment_adjusted"]],
    v2[["month", "sentiment_adjusted_2"]],
    on="month"
)

# --- LÍNEAS DE TENDENCIA ---
x = np.arange(len(merged))
y_v1 = merged["sentiment_adjusted"].values
y_v2 = merged["sentiment_adjusted_2"].values
m1, b1 = np.polyfit(x, y_v1, 1)
m2, b2 = np.polyfit(x, y_v2, 1)

# --- GRAFICAR ---
plt.figure(figsize=(12, 5))

# Curvas de sentimiento
plt.plot(merged["month"], y_v1, label="Versión 1 (solo P10)", color='orange')
plt.plot(merged["month"], y_v2, label="Versión 2 (P10 y P90)", color='green')

# Tendencias
plt.plot(merged["month"], m1 * x + b1, '--', color='darkorange', label="Tendencia v1")
plt.plot(merged["month"], m2 * x + b2, '--', color='darkgreen', label="Tendencia v2")

# Líneas verticales en meses ajustados
for fecha in meses_ajustados:
    plt.axvline(x=fecha, color='gray', linestyle=':', alpha=0.5)

# Estética
plt.title("Comparación entre sentimiento ajustado v1 y v2 (con tendencia y ajustes)")
plt.xlabel("Fecha")
plt.ylabel("Sentimiento (VADER)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar imagen
output_path = os.path.join(output_folder, "comparacion_sentimiento_v1_v2_con_tendencia_y_ajustes.png")
plt.savefig(output_path)
plt.close()

print("Gráfica guardada en:", output_path)
