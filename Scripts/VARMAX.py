# ====================================================
# Modelo VARMAX entre SP500 y sentimiento
# ====================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ===============================
# 1. Cargar y preparar datos
# ===============================

# Cargar S&P 500
df_sp = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Finance\sp500_10_anios.xlsx")
df_sp['Fecha'] = pd.to_datetime(df_sp['Fecha'], dayfirst=True)
df_sp.set_index('Fecha', inplace=True)
df_sp_mensual = df_sp.resample('ME').last().copy()
df_sp_mensual.rename(columns={'Cierre': 'sp500'}, inplace=True)
df_sp_mensual.index = df_sp_mensual.index.to_period("M").to_timestamp()

# Cargar sentimiento
df_sent = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted.xlsx")
df_sent['month'] = pd.to_datetime(df_sent['month'])
df_sent.set_index('month', inplace=True)
df_sent.rename(columns={'sentiment_adjusted': 'sentimiento'}, inplace=True)
df_sent.index = df_sent.index.to_period("M").to_timestamp()

# Unir y filtrar
df = df_sp_mensual.join(df_sent, how='inner')
df = df[['sp500', 'sentimiento']].dropna()

# ===============================
# 2. Diferenciación (si no estacionaria)
# ===============================

df_diff = df.diff().dropna()

# ===============================
# 3. Ajuste del modelo VARMAX
# ===============================

modelo = VARMAX(df_diff, order=(1, 0))
resultado = modelo.fit(disp=False)
print(resultado.summary())

# ===============================
# 4. Predicción in-sample
# ===============================

pred = resultado.predict(start=0, end=len(df_diff)-1)
pred_cumsum = pred.cumsum() + df.iloc[0]

# ===============================
# 5. Evaluación
# ===============================

real = df.loc[pred_cumsum.index]
rmse = np.sqrt(mean_squared_error(real['sp500'], pred_cumsum['sp500']))
mae = mean_absolute_error(real['sp500'], pred_cumsum['sp500'])
mape = np.mean(np.abs((real['sp500'] - pred_cumsum['sp500']) / real['sp500'])) * 100

print(f"RMSE VARMAX: {rmse:.2f}")
print(f"MAE VARMAX: {mae:.2f}")
print(f"MAPE VARMAX: {mape:.2f}%")

# ===============================
# 6. Visualización
# ===============================

plt.figure(figsize=(10, 5))
plt.plot(real['sp500'], label='Real SP500')
plt.plot(pred_cumsum['sp500'], label='Predicción VARMAX', linestyle='--')
plt.title("Predicción SP500 con VARMAX")
plt.legend()
plt.tight_layout()
plt.show()
