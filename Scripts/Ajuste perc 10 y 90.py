import pandas as pd

# Cargar el archivo original
df_sent = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted.xlsx")
df_sent['month'] = pd.to_datetime(df_sent['month'])
df_sent.set_index('month', inplace=True)

# Calcular percentiles y media
p10 = df_sent['comment_count'].quantile(0.10)
p90 = df_sent['comment_count'].quantile(0.90)
media_sent = df_sent['sentiment_adjusted'].mean()

# Crear columna ajustada
df_sent['sentiment_adjusted_2'] = df_sent['sentiment_adjusted']

# Reemplazar valores fuera de los percentiles
df_sent.loc[
    (df_sent['comment_count'] < p10) | (df_sent['comment_count'] > p90),
    'sentiment_adjusted_2'
] = media_sent

# Guardar nuevo archivo
df_sent.to_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted_v2.xlsx")

print("Archivo guardado como 'sentiment_adjusted_v2.xlsx'")
