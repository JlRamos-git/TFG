# --- IMPORTACIONES ---
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import math
import os

# --- CONFIGURACIÓN ---
input_path = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\reddit_cleaned_filtered.csv"
output_path = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\nube_compuesta.png"

# --- LECTURA DEL DATASET ---
df = pd.read_csv(input_path)
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)

# --- OBTENER TEXTOS AGRUPADOS POR MES ---
textos_por_mes = df.groupby("month")["clean_text"].apply(lambda x: " ".join(x.dropna()))

# --- CONFIGURAR LA CUADRÍCULA ---
num_meses = len(textos_por_mes)
cols = 3
rows = math.ceil(num_meses / cols)
fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
axes = axes.flatten()

# --- GENERAR NUBES DE PALABRAS ---
for i, (mes, texto) in enumerate(textos_por_mes.items()):
    if texto.strip():
        wc = WordCloud(width=800, height=400, background_color='white').generate(texto)
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(mes, fontsize=14)
    else:
        axes[i].axis('off')
        axes[i].set_title(f"{mes} (vacío)", fontsize=12, color='gray')

# --- OCULTAR PANELES VACÍOS ---
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()
print(f" Nube compuesta guardada en: {output_path}")
