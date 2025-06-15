# ====================================================
# Cálculo de MAE y MAPE para ARIMA vs SARIMAX
# ====================================================

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

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

# Unir datasets por índice
df = df_sp_mensual.join(df_sent[['sentimiento']], how='inner')
df.dropna(inplace=True)

endog = df['sp500']
exog = df[['sentimiento']]

# ===============================
# 2. Modelo ARIMA(1,1,1)
# ===============================

modelo_arima = ARIMA(endog, order=(1,1,1)).fit()
pred_arima = modelo_arima.predict(start=endog.index[1], end=endog.index[-1])
real_arima = endog[pred_arima.index]

mae_arima = mean_absolute_error(real_arima, pred_arima)
mape_arima = np.mean(np.abs((real_arima - pred_arima) / real_arima)) * 100

# ===============================
# 3. Modelo SARIMAX(2,1,2)
# ===============================

modelo_sarimax = SARIMAX(endog, exog=exog, order=(2,1,2),
                         enforce_stationarity=False, enforce_invertibility=False)
resultado_sarimax = modelo_sarimax.fit(disp=False)
pred_sarimax = resultado_sarimax.predict(start=endog.index[1], end=endog.index[-1], exog=exog)
real_sarimax = endog[pred_sarimax.index]

mae_sarimax = mean_absolute_error(real_sarimax, pred_sarimax)
mape_sarimax = np.mean(np.abs((real_sarimax - pred_sarimax) / real_sarimax)) * 100

# ===============================
# 4. Resultados
# ===============================

print("Comparativa de métricas de error:")
print(f"ARIMA(1,1,1)   → MAE: {mae_arima:.2f} | MAPE: {mape_arima:.2f}%")
print(f"SARIMAX(2,1,2) → MAE: {mae_sarimax:.2f} | MAPE: {mape_sarimax:.2f}%")

# ===============================
# 5. Visualización de residuos
# ===============================

residuos_arima = real_arima - pred_arima
residuos_sarimax = real_sarimax - pred_sarimax

plt.figure(figsize=(10, 5))
plt.plot(residuos_arima, label='Residuos ARIMA', linestyle='--')
plt.plot(residuos_sarimax, label='Residuos SARIMAX', linestyle=':')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Residuos de los modelos ARIMA y SARIMAX")
plt.legend()
plt.tight_layout()
plt.show()
