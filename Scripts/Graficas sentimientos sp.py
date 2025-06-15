import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURACIÓN DE RUTAS ---
ruta_sp500 = r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Finance\sp500_10_anios.xlsx"
ruta_sentimiento = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted.xlsx"
output_folder = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\Graficas_Exploratorias"
os.makedirs(output_folder, exist_ok=True)

# --- CARGA DE DATOS ---
sp500 = pd.read_excel(ruta_sp500)
sentimiento = pd.read_excel(ruta_sentimiento)

# --- FORMATO DE FECHAS ---
sp500["Fecha"] = pd.to_datetime(sp500["Fecha"], dayfirst=True)
sentimiento["month"] = pd.to_datetime(sentimiento["month"])

# --- MERGE ---
merged = pd.merge(sp500, sentimiento, left_on="Fecha", right_on="month")

# --- 1. Evolución del S&P 500 ---
plt.figure(figsize=(12, 5))
plt.plot(sp500["Fecha"], sp500["Cierre"], label="S&P 500", color='blue')
plt.title("Evolución mensual del S&P 500 (2015–2025)")
plt.xlabel("Fecha")
plt.ylabel("Cierre")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "evolucion_sp500.png"))
plt.close()

# --- 2. Evolución del sentimiento ajustado ---
plt.figure(figsize=(12, 5))
plt.plot(sentimiento["month"], sentimiento["sentiment_adjusted"], label="Sentimiento Ajustado", color='orange')
plt.title("Evolución mensual del sentimiento económico ajustado (VADER)")
plt.xlabel("Fecha")
plt.ylabel("Sentimiento (VADER)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "evolucion_sentimiento_ajustado.png"))
plt.close()

# --- 3. Comparación conjunta ---
fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.set_xlabel("Fecha")
ax1.set_ylabel("S&P 500", color='blue')
ax1.plot(merged["Fecha"], merged["Cierre"], color='blue', label="S&P 500")
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Sentimiento ajustado", color='red')
ax2.plot(merged["Fecha"], merged["sentiment_adjusted"], color='red', label="Sentimiento")
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Comparación mensual entre S&P 500 y sentimiento económico ajustado")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "comparacion_sp500_sentimiento_ajustado.png"))
plt.close()

# --- 4. Correlación entre sentimiento y S&P 500 ---
plt.figure(figsize=(8, 6))
sns.regplot(x=merged["sentiment_adjusted"], y=merged["Cierre"])
plt.title("Correlación entre sentimiento económico ajustado y S&P 500")
plt.xlabel("Sentimiento (VADER)")
plt.ylabel("Cierre del S&P 500")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "correlacion_sentimiento_sp500_ajustado.png"))
plt.close()

print("✅ Gráficas generadas correctamente en:")
print(output_folder)
