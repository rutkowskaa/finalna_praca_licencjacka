import yfinance as yf
import pandas as pd

class Get_Data:
    def __init__(self, nazwa_instrumentu:str, start:str, interval:str, days_train:int=30):
        print("Konstruktor")
        self.make_diff()
        self.analiza_statystyczna_szeregu(pd.Series)

    def make_diff(self):
        print("make_diff()")

    @staticmethod
    def analiza_statystyczna_szeregu(szereg_pandas: pd.Series, max_lag:int = 30, co_sprawdzamy:str="DANE PODSTAWOWE", wykres:bool=True, crit:str="AIC"):
        print("analiza_statystyczna_szeregu")

getter = Get_Data("asd", "asd", "asd", "123")
