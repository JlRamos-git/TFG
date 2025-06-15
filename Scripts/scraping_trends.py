import os
import pandas as pd

# --- RUTA DE DESCARGAS ---
ruta_descargas = os.path.join(os.path.expanduser("~"), "Downloads")
archivos = [f for f in os.listdir(ruta_descargas) if f.startswith("trend_") and f.endswith(".csv")]

if not archivos:
    print("No se encontraron archivos 'trend_*.csv' en la carpeta de Descargas.")
    exit()

dataframes = []

for archivo in archivos:
    nombre_subreddit = archivo.replace("trend_", "").replace(".csv", "")
    ruta_completa = os.path.join(ruta_descargas, archivo)

    try:
        # LEER CON SEPARADOR CORRECTO Y SALTAR FILAS INICIALES
        df = pd.read_csv(
            ruta_completa,
            skiprows=3,              # saltar categoría, línea vacía y encabezado
            sep=",",                 # separador real
            header=None,             # no usar encabezado automático
            names=["date", "value"], # asignar nombres
            encoding="utf-8"
        )

        # CONVERSIÓN DE TIPOS
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m", errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["subreddit"] = nombre_subreddit

        print(f"Cargado correctamente: {archivo}")
        print(df.head(3))

        dataframes.append(df)

    except Exception as e:
        print(f"Error al procesar {archivo}: {e}")

# --- UNIFICAR Y GUARDAR ---
if dataframes:
    df_final = pd.concat(dataframes, ignore_index=True)

    # Convertir columna 'value' a entero
    df_final["value"] = df_final["value"].astype("Int64")

    ruta_salida = r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Processed\google_trends_unificado.csv"
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df_final.to_csv(ruta_salida, index=False)
    print(f"\n CSV unificado guardado en: {ruta_salida}")
else:
    print("No se pudieron procesar archivos válidos.")
