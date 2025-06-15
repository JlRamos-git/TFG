# ====================================================
# MODELO SARIMAX CON VARIABLE EXÓGENA: SENTIMIENTO
# ====================================================

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ===============================
# 1. CARGAR SERIES NECESARIAS
# ===============================

# Cargar el S&P 500 mensual desde Excel
df_sp = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Finance\sp500_10_anios.xlsx")
df_sp['Fecha'] = pd.to_datetime(df_sp['Fecha'], dayfirst=True)
df_sp.set_index('Fecha', inplace=True)

# Reescalar a frecuencia mensual (último día del mes) y renombrar la columna de interés
df_sp_mensual = df_sp.resample('ME').last().copy()
df_sp_mensual.rename(columns={'Cierre': 'sp500'}, inplace=True)

# Cargar el sentimiento ajustado desde Excel
df_sent = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted.xlsx")
df_sent['month'] = pd.to_datetime(df_sent['month'])
df_sent.set_index('month', inplace=True)
df_sent.rename(columns={'sentiment_adjusted': 'sentimiento'}, inplace=True)

# ===============================
# 2. ALINEAR ÍNDICES TEMPORALES
# ===============================

# Convertir ambos índices a primer día del mes para garantizar coincidencia
df_sp_mensual.index = df_sp_mensual.index.to_period("M").to_timestamp()
df_sent.index = df_sent.index.to_period("M").to_timestamp()

# ===============================
# 3. UNIR LAS DOS SERIES
# ===============================

# Unimos por índice temporal común (primer día del mes)
df = df_sp_mensual.join(df_sent[['sentimiento']], how='inner')

# Eliminamos cualquier fila con nulos (por seguridad)
df.dropna(inplace=True)

# Validamos que hay datos
print(df.head())
print("Forma final del dataframe:", df.shape)

# ===============================
# 4. DEFINIR VARIABLES PARA EL MODELO
# ===============================

endog = df['sp500']             # Variable endógena: índice S&P 500
exog = df[['sentimiento']]      # Variable exógena: sentimiento mensual

# ===============================
# 5. AJUSTE DEL MODELO SARIMAX
# ===============================

# Ajustamos un SARIMAX simple con (p,d,q) = (1,1,1)
modelo_sarimax = SARIMAX(endog, exog=exog, order=(1,1,1),
                         enforce_stationarity=False, enforce_invertibility=False)

resultado_sarimax = modelo_sarimax.fit(disp=False)

# ===============================
# 6. EVALUACIÓN DEL MODELO
# ===============================

# Mostramos resumen estadístico del modelo
print(resultado_sarimax.summary())

# ===============================
# 7. PREDICCIÓN DENTRO DE LA MUESTRA
# ===============================

# Predicción sobre el mismo periodo de entrenamiento
pred_sarimax = resultado_sarimax.predict(start=endog.index[1],
                                         end=endog.index[-1],
                                         exog=exog)

# ===============================
# 8. VISUALIZACIÓN DEL AJUSTE
# ===============================

plt.figure(figsize=(10, 5))
plt.plot(endog, label='S&P 500 real')
plt.plot(pred_sarimax, label='SARIMAX (con sentimiento)', linestyle='--')
plt.legend()
plt.title("Modelo SARIMAX ajustado con variable exógena de sentimiento")
plt.tight_layout()
plt.show()

# ===============================
# 9. GUARDAR RESULTADOS EN EXCEL
# ===============================

df_resultado = pd.DataFrame({
    'month': pred_sarimax.index,
    'real': endog[pred_sarimax.index].values,
    'prediction': pred_sarimax.values
})
df_resultado.to_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sarimax_sentimiento_v1.xlsx", index=False)
