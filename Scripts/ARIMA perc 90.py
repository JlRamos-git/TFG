# ====================================================
# ARIMA(1,1,1) tras excluir meses con volumen alto
# ====================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ===============================
# 1. Cargar datos
# ===============================

# SP500
df_sp = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Finance\sp500_10_anios.xlsx")
df_sp['Fecha'] = pd.to_datetime(df_sp['Fecha'], dayfirst=True)
df_sp.set_index('Fecha', inplace=True)
df_sp_mensual = df_sp.resample('ME').last().copy()
df_sp_mensual.rename(columns={'Cierre': 'sp500'}, inplace=True)
df_sp_mensual.index = df_sp_mensual.index.to_period("M").to_timestamp()

# Sentimiento + comentarios
df_sent = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted.xlsx")
df_sent['month'] = pd.to_datetime(df_sent['month'])
df_sent.set_index('month', inplace=True)
df_sent.rename(columns={'sentiment_adjusted': 'sentimiento'}, inplace=True)
df_sent.index = df_sent.index.to_period("M").to_timestamp()

# ===============================
# 2. Filtrado por percentil 90
# ===============================

umbral_90 = df_sent['comment_count'].quantile(0.90)
df_sent_filtrado = df_sent[df_sent['comment_count'] <= umbral_90]

# ===============================
# 3. Unir datasets y preparar serie
# ===============================

df_comb = df_sp_mensual.join(df_sent_filtrado[['sentimiento']], how='inner')
df_comb.dropna(inplace=True)
serie = df_comb['sp500']

# ===============================
# 4. ARIMA(1,1,1)
# ===============================

modelo = ARIMA(serie, order=(1, 1, 1)).fit()
print(modelo.summary())

# ===============================
# 5. Predicción y evaluación
# ===============================

pred = modelo.predict(start=serie.index[1], end=serie.index[-1])
real = serie[pred.index]

rmse = np.sqrt(mean_squared_error(real, pred))
mae = mean_absolute_error(real, pred)
mape = np.mean(np.abs((real - pred) / real)) * 100

print(f"RMSE (filtrado): {rmse:.2f}")
print(f"MAE  (filtrado): {mae:.2f}")
print(f"MAPE (filtrado): {mape:.2f}%")

# ===============================
# 6. Visualización
# ===============================

plt.figure(figsize=(10, 5))
plt.plot(serie, label='SP500 real')
plt.plot(pred, label='Predicción ARIMA filtrado', linestyle='--')
plt.title("ARIMA(1,1,1) con meses filtrados por volumen")
plt.legend()
plt.tight_layout()
plt.show()
