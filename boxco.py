import resources.Get_Data as Get_Data
import yfinance
from resources.single_data.RF_AR import RF_AR
import matplotlib.pyplot as plt
import pandas as pd

data = Get_Data.Get_Data(nazwa_instrumentu="^IXIC", start='2020-01-01', interval='1d')
close = data.make_norm_diff()

print(close)