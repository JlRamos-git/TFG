# ============================================
# SARIMAX(2,1,2) con sentimiento ajustado v2.0
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========================
# 1. Cargar los datos
# ========================

# SP500 mensual
df_sp = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Finance\sp500_10_anios.xlsx")
df_sp['Fecha'] = pd.to_datetime(df_sp['Fecha'], dayfirst=True)
df_sp.set_index('Fecha', inplace=True)
df_sp_mensual = df_sp.resample('M').last().copy()
df_sp_mensual.rename(columns={'Cierre': 'sp500'}, inplace=True)
df_sp_mensual.index = df_sp_mensual.index.to_period("M").to_timestamp()

# Sentimiento ajustado v2
df_sent = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted_v2.xlsx")
df_sent['month'] = pd.to_datetime(df_sent['month'])
df_sent.set_index('month', inplace=True)
df_sent.index = df_sent.index.to_period("M").to_timestamp()
df_sent.rename(columns={'sentiment_adjusted_2': 'sentimiento'}, inplace=True)

# ========================
# 2. Combinar y preparar
# ========================

df = df_sp_mensual.join(df_sent[['sentimiento']], how='inner')
df.dropna(inplace=True)

endog = df['sp500']
exog = df[['sentimiento']]

# ========================
# 3. Modelo SARIMAX(2,1,2)
# ========================

modelo_sarimax = SARIMAX(endog, exog=exog, order=(2,1,2), enforce_stationarity=False, enforce_invertibility=False).fit()

print(modelo_sarimax.summary())

# ========================
# 4. Evaluación
# ========================

pred = modelo_sarimax.predict(start=endog.index[1], end=endog.index[-1], exog=exog.iloc[1:])
real = endog[pred.index]

rmse = np.sqrt(mean_squared_error(real, pred))
mae = mean_absolute_error(real, pred)
mape = np.mean(np.abs((real - pred) / real)) * 100
aic = modelo_sarimax.aic

print(f"\nAIC (SARIMAX v2): {aic:.2f}")
print(f"RMSE (SARIMAX v2): {rmse:.2f}")
print(f"MAE  (SARIMAX v2): {mae:.2f}")
print(f"MAPE (SARIMAX v2): {mape:.2f}%")


# ========================
# 5. Visualización
# ========================

plt.figure(figsize=(10, 5))
plt.plot(endog, label="SP500 real")
plt.plot(pred, label="Predicción SARIMAX(2,1,2)", linestyle="--")
plt.title("Modelo SARIMAX(2,1,2) con sentimiento ajustado v2.0")
plt.legend()
plt.tight_layout()
plt.show()
