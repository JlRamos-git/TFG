# --- IMPORTACIONES ---
import pandas as pd
import time
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
import os
from nltk.corpus import stopwords
nltk.download("stopwords")
from wordcloud import WordCloud 
stopwords = set(stopwords.words('spanish', 'english'))
# --- CONFIGURACIÓN ---
input_path = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\reddit_cleaned_filtered.csv"
output_folder = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\nubes_mensuales"
os.makedirs(output_folder, exist_ok=True)

# --- LECTURA DEL DATASET ---
df = pd.read_csv(input_path)
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)

# --- GENERAR NUBE DE PALABRAS POR MES ---
for month in df["month"].unique():
    subset = df[df["month"] == month]
    if subset.empty:
        continue

    text = " ".join(subset["clean_text"].dropna())
    if not text.strip():
        continue

    wordcloud = WordCloud(width=1000, height=400, background_color='white', stopwords= stopwords).generate(text)
    plt.figure(figsize=(12, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Nube de Palabras - {month}", fontsize=16)
    plt.tight_layout()

    output_path = os.path.join(output_folder, f"nube_{month}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Nube guardada para {month} en {output_path}")
