# =============================================
# VARMAX con sentimiento ajustado v2.0
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
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
df = df[['sp500', 'sentimiento']].dropna()

# ========================
# 3. Diferenciación
# ========================

df_diff = df.diff().dropna()

# ========================
# 4. Ajuste del modelo VARMAX
# ========================

modelo = VARMAX(df_diff, order=(1, 0))
resultado = modelo.fit(disp=False)
print(resultado.summary())

# ========================
# 5. Predicción y evaluación
# ========================

pred = resultado.predict(start=0, end=len(df_diff)-1)
pred_cumsum = pred.cumsum() + df.iloc[0]

real = df.loc[pred_cumsum.index]
rmse = np.sqrt(mean_squared_error(real['sp500'], pred_cumsum['sp500']))
mae = mean_absolute_error(real['sp500'], pred_cumsum['sp500'])
mape = np.mean(np.abs((real['sp500'] - pred_cumsum['sp500']) / real['sp500'])) * 100

print(f"RMSE VARMAX v2: {rmse:.2f}")
print(f"MAE  VARMAX v2: {mae:.2f}")
print(f"MAPE VARMAX v2: {mape:.2f}%")

# ========================
# 6. Visualización
# ========================

plt.figure(figsize=(10, 5))
plt.plot(real['sp500'], label='SP500 real')
plt.plot(pred_cumsum['sp500'], label='Predicción VARMAX v2', linestyle='--')
plt.title("VARMAX con sentimiento ajustado v2.0")
plt.legend()
plt.tight_layout()
plt.show()
