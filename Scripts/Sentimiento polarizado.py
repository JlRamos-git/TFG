import pandas as pd

# Ruta del archivo de sentimiento mensual
ruta_csv = r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_monthly.csv"

# Leer CSV con separador punto y coma
df = pd.read_csv(ruta_csv, sep=';')

# Convertir 'month' a datetime
df['month'] = pd.to_datetime(df['month'])

# Ordenar cronológicamente
df = df.sort_values('month')

# Calcular percentil 10 del número de comentarios
percentil_10 = df['comment_count'].quantile(0.10)

# Calcular la media global del sentimiento
media_sentimiento = df['sentiment_mean'].mean()

# Crear nueva columna: si el número de comentarios < P10 => usar la media
df['sentiment_adjusted'] = df.apply(
    lambda row: row['sentiment_mean'] if row['comment_count'] >= percentil_10 else media_sentimiento,
    axis=1
)

# Vista previa
print(df[['month', 'sentiment_mean', 'comment_count', 'sentiment_adjusted']].head())

# Guardar si quieres
# df.to_csv(r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted.csv", index=False)
