# --- IMPORTACIONES ---
import pandas as pd
import nltk
import re
import os

# --- DESCARGA DE STOPWORDS ---
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# --- DEFINICIÓN DE PALABRAS CLAVE ECONÓMICAS ---
economic_terms = ["economy", "inflation", "recession", "crisis", "tariff", "interest", "gdp", "unemployment"]

# --- CARGAR DATOS ---
input_path = r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Reddit\reddit_comments_10years_2025-05-12.csv" #  Ajusta la fecha del archivo
output_path = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\reddit_cleaned_filtered.csv"

# --- FUNCIONES AUXILIARES ---
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()

# --- LECTURA DEL DATASET Y LIMPIEZA ---
df = pd.read_csv(input_path)
df["date"] = pd.to_datetime(df["date"])
df["clean_text"] = df["clean_text"].astype(str).apply(clean_text)

# --- FILTRADO POR TÉRMINOS ECONÓMICOS ---
df = df[df["clean_text"].str.contains('|'.join(economic_terms), case=False, na=False)]

# --- GUARDAR RESULTADO LIMPIO Y FILTRADO ---
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f" Dataset depurado y filtrado guardado en: {output_path}")
