# ====================================================
# Comparación de modelos SARIMAX con sentimiento exógeno
# ====================================================

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ===============================
# 1. Cargar series preparadas
# ===============================

# Cargar S&P 500 mensual
df_sp = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Finance\sp500_10_anios.xlsx")
df_sp['Fecha'] = pd.to_datetime(df_sp['Fecha'], dayfirst=True)
df_sp.set_index('Fecha', inplace=True)
df_sp_mensual = df_sp.resample('ME').last().copy()
df_sp_mensual.rename(columns={'Cierre': 'sp500'}, inplace=True)
df_sp_mensual.index = df_sp_mensual.index.to_period("M").to_timestamp()

# Cargar sentimiento mensual desde Excel
df_sent = pd.read_excel(r"C:\Users\gonlo\Desktop\TFG\Datos\Resultados\sentiment_adjusted.xlsx")
df_sent['month'] = pd.to_datetime(df_sent['month'])
df_sent.set_index('month', inplace=True)
df_sent.rename(columns={'sentiment_adjusted': 'sentimiento'}, inplace=True)
df_sent.index = df_sent.index.to_period("M").to_timestamp()

# ===============================
# 2. Unir y preparar variables
# ===============================

df = df_sp_mensual.join(df_sent[['sentimiento']], how='inner')
df.dropna(inplace=True)
endog = df['sp500']
exog = df[['sentimiento']]

# ===============================
# 3. Evaluar distintos modelos SARIMAX
# ===============================

ordenes = [(0,1,0), (1,1,0), (0,1,1), (1,1,1), (1,1,2), (2,1,2)]

resultados = []

for order in ordenes:
    try:
        modelo = SARIMAX(endog, exog=exog, order=order,
                         enforce_stationarity=False, enforce_invertibility=False)
        resultado = modelo.fit(disp=False)

        coef = resultado.params.get('sentimiento', None)
        pval = resultado.pvalues.get('sentimiento', None)
        aic = resultado.aic

        resultados.append((order, aic, coef, pval))

    except Exception as e:
        print(f"Error con modelo SARIMAX{order}: {e}")

# ===============================
# 4. Mostrar tabla comparativa
# ===============================

print(f"{'Modelo':<15} {'AIC':<10} {'Coef. sentimiento':<20} {'p-valor':<10}")
print("-"*60)
for order, aic, coef, pval in resultados:
    print(f"SARIMAX{order}  {aic:<10.2f} {coef:<20.2f} {pval:<10.3f}")
