# --- IMPORTACIONES ---
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# --- DESCARGA DE LEXICO VADER ---
nltk.download('vader_lexicon')

# --- CONFIGURACIÓN DE RUTAS ---
input_path = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\reddit_cleaned_filtered.csv"
output_path = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_monthly.csv"

# --- CARGA DE DATOS ---
df = pd.read_csv(input_path)
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)

# --- LIMPIEZA DE TEXTOS VACÍOS O NULOS ---
df = df[df["clean_text"].notna() & (df["clean_text"].str.strip() != "")].copy()

# --- ASEGURAR FORMATO DE TEXTO ---
df["clean_text"] = df["clean_text"].astype(str)

# --- ANALISIS DE SENTIMIENTO ---
sia = SentimentIntensityAnalyzer()
df["compound"] = df["clean_text"].apply(lambda x: sia.polarity_scores(x)["compound"])

# --- ASEGURAR FORMATO NUMÉRICO ---
df["compound"] = pd.to_numeric(df["compound"], errors="coerce")

# --- AGRUPACION MENSUAL ---
monthly_sentiment = df.groupby("month")["compound"].mean().reset_index()
monthly_sentiment = monthly_sentiment.rename(columns={"compound": "sentiment_mean"})
monthly_sentiment["comment_count"] = df.groupby("month")["clean_text"].count().values

# --- GUARDAR RESULTADO ---
os.makedirs(os.path.dirname(output_path), exist_ok=True)
monthly_sentiment["sentiment_mean"] = monthly_sentiment["sentiment_mean"].round(4)
monthly_sentiment.to_csv(output_path, index=False, sep=';', float_format='%.4f')
print(f" Sentimiento mensual guardado en: {output_path}")

