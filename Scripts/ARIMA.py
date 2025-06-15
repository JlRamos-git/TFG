# ====================================================
# 1. Cargar y diferenciar la serie del S&P 500
# ====================================================

import pandas as pd
import matplotlib.pyplot as plt

# Cargar archivo mensual (debe estar previamente generado)
df_sp = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Finance\sp500_10_anios.xlsx")
df_sp['Fecha'] = pd.to_datetime(df_sp['Fecha'], dayfirst=True)
df_sp.set_index('Fecha', inplace=True)
df_mensual = df_sp.resample('ME').last().copy()
df_mensual.rename(columns={'Cierre': 'sp500'}, inplace=True)

# Diferenciación (d=1)
serie = df_mensual['sp500'].dropna()
serie_diff = serie.diff().dropna()

# Visualizar la serie diferenciada
plt.figure(figsize=(10, 4))
plt.plot(serie_diff, label='Serie diferenciada (d=1)')
plt.title("S&P 500 diferenciado (1ª orden)")
plt.legend()
plt.show()


# ====================================================
# 2. Análisis ACF y PACF
# ====================================================

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(serie_diff, lags=24)
plot_pacf(serie_diff, lags=24)
plt.show()


# ====================================================
# 3. Búsqueda manual de parámetros (p,d,q) por AIC
# ====================================================

from statsmodels.tsa.arima.model import ARIMA

resultados = []
for p in range(3):
    for q in range(3):
        try:
            modelo = ARIMA(serie, order=(p,1,q)).fit()
            resultados.append((p, 1, q, modelo.aic))
        except:
            continue

# Ordenar resultados por menor AIC
resultados = sorted(resultados, key=lambda x: x[3])
for r in resultados:
    print(f"ARIMA({r[0]},{r[1]},{r[2]}) - AIC: {r[3]:.2f}")


# ====================================================
# 4. Ajuste del mejor modelo ARIMA y diagnóstico
# ====================================================

# ARIMA(1,1,1)
modelo = ARIMA(serie, order=(1,1,1)).fit()

# Diagnóstico de residuos
modelo.plot_diagnostics(figsize=(10, 6))
plt.show()


# ====================================================
# 5. Predicción dentro de la muestra (in-sample)
# ====================================================

pred = modelo.predict(start=serie.index[1], end=serie.index[-1])

plt.figure(figsize=(10, 5))
plt.plot(serie, label='Serie real')
plt.plot(pred, label='Predicción ARIMA (in-sample)', linestyle='--')
plt.legend()
plt.title("Predicción ARIMA ajustada sobre el S&P 500")
plt.show()
