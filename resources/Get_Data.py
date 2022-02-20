import yfinance as yf
import pandas as pd

class Get_Data:
    def __init__(self, nazwa_instrumentu:str, start, interval:str):
        """
        :param nazwa_instrumentu: Kod instrumentu zgodny z yahoo finance
        :param start: początek danych
        :param interval: interwał brany pod uwagę
        :param days_train:
        """
        tick = yf.Ticker(nazwa_instrumentu)
        self.dane = tick.history(start=start, interval=interval)

    def make_diff(self):
        print("make_diff()")
        result = self.dane["Close"].diff(1)[1:]
        return result

    @staticmethod
    def analiza_statystyczna_szeregu(szereg_pandas: pd.Series, max_lag:int = 30, co_sprawdzamy:str="DANE PODSTAWOWE", wykres:bool=True, crit:str="AIC"):
        print("analiza_statystyczna_szeregu")



getter = Get_Data(nazwa_instrumentu="^IXIC", start="2021-01-01", interval="1d")
