# ====================================================
# Comparación ARIMA vs SARIMAX con RMSE
# ====================================================

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ===============================
# 1. Cargar y preparar datos
# ===============================

# Cargar S&P 500 mensual
df_sp = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Finance\sp500_10_anios.xlsx")
df_sp['Fecha'] = pd.to_datetime(df_sp['Fecha'], dayfirst=True)
df_sp.set_index('Fecha', inplace=True)
df_sp_mensual = df_sp.resample('ME').last().copy()
df_sp_mensual.rename(columns={'Cierre': 'sp500'}, inplace=True)
df_sp_mensual.index = df_sp_mensual.index.to_period("M").to_timestamp()

# Cargar sentimiento mensual ajustado
df_sent = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted.xlsx")
df_sent['month'] = pd.to_datetime(df_sent['month'])
df_sent.set_index('month', inplace=True)
df_sent.rename(columns={'sentiment_adjusted': 'sentimiento'}, inplace=True)
df_sent.index = df_sent.index.to_period("M").to_timestamp()

# Unir ambos datasets por índice mensual
df = df_sp_mensual.join(df_sent[['sentimiento']], how='inner')
df.dropna(inplace=True)

# Definir series
endog = df['sp500']
exog = df[['sentimiento']]

# ===============================
# 2. Modelo ARIMA(1,1,1)
# ===============================

modelo_arima = ARIMA(endog, order=(1,1,1)).fit()
pred_arima = modelo_arima.predict(start=endog.index[1], end=endog.index[-1])
real_arima = endog[pred_arima.index]
rmse_arima = np.sqrt(mean_squared_error(real_arima, pred_arima))

# ===============================
# 3. Modelo SARIMAX(2,1,2)
# ===============================

modelo_sarimax = SARIMAX(endog, exog=exog, order=(2,1,2),
                         enforce_stationarity=False, enforce_invertibility=False)
resultado_sarimax = modelo_sarimax.fit(disp=False)
pred_sarimax = resultado_sarimax.predict(start=endog.index[1], end=endog.index[-1], exog=exog)
real_sarimax = endog[pred_sarimax.index]
rmse_sarimax = np.sqrt(mean_squared_error(real_sarimax, pred_sarimax))

# ===============================
# 4. Resultados comparativos
# ===============================

print("Comparativa de RMSE entre modelos:")
print(f"ARIMA(1,1,1):   {rmse_arima:.2f}")
print(f"SARIMAX(2,1,2): {rmse_sarimax:.2f}")


import pandas as pd

# Cargar el archivo del S&P 500
df_sp = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Finance\sp500_10_anios.xlsx")
df_sp['Fecha'] = pd.to_datetime(df_sp['Fecha'], dayfirst=True)
df_sp.set_index('Fecha', inplace=True)

# Resamplear a mensual y calcular la media
df_sp_mensual = df_sp.resample('ME').last().copy()
media_sp500 = df_sp_mensual['Cierre'].mean()

# Usar el RMSE que obtuviste para ARIMA
rmse_arima = 166.59
porcentaje_error = (rmse_arima / media_sp500) * 100

print(f"Media del S&P 500 mensual: {media_sp500:.2f}")
print(f"Porcentaje de RMSE respecto a la media: {porcentaje_error:.2f}%")
