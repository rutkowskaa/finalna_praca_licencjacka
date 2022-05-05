import resources.Get_Data as Get_Data
import yfinance
from resources.single_data.RF_AR import RF_AR
import matplotlib.pyplot as plt
import pandas as pd

data = Get_Data.Get_Data(nazwa_instrumentu="^GSPC", start='2022-04-01', interval='5m')
close = data.make_diff()

print(Get_Data.Get_Data.analiza_statystyczna_szeregu(close))
plt.scatter(close[1:], close.shift(1)[1:])
plt.show()