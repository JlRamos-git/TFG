import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Fechas
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 10)

# Descargar datos
sp500 = yf.download('^GSPC', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=False)

# Filtrar y renombrar columnas
sp500 = sp500[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
sp500.reset_index(inplace=True)

# Formatear fecha
sp500['Date'] = sp500['Date'].dt.strftime('%d/%m/%Y')

# Eliminar filas basura tipo ^GSPC
sp500 = sp500[~sp500.astype(str).apply(lambda row: row.str.contains(r'\^GSPC')).any(axis=1)]

# RECONSTRUIR DataFrame para eliminar cualquier MultiIndex oculto
sp500 = pd.DataFrame(sp500.values, columns=['Fecha', 'Apertura', 'Máximo', 'Mínimo', 'Cierre', 'Cierre_Ajustado', 'Volumen'])

# Exportar a Excel
output_path = r'C:\Users\gonlo\Desktop\TFG\Datos\Raw\Finance\sp500_10_anios.xlsx'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
sp500.to_excel(output_path, index=False)

print(f"✅ Excel limpio generado en:\n{output_path}")
