import scipy
import yfinance as yf
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, pacf, acf, kpss
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import bds

class Get_Data:
    def __init__(self, nazwa_instrumentu:str, start, interval:str, end=None):
        """
        :param nazwa_instrumentu: Kod instrumentu zgodny z yahoo finance
        :param start: początek danych
        :param interval: interwał brany pod uwagę
        :param days_train:
        """
        tick = yf.Ticker(nazwa_instrumentu)
        if end == None:
            self.dane = tick.history(start=start, interval=interval)
        else:
            self.dane = tick.history(start=start, end=end, interval=interval)
    def make_diff(self):
        print("make_diff()")
        result = self.dane["Close"].diff(1)[1:]
        return result

    def make_log_diff(self):
        print("make_diff()")
        result = np.log(self.dane["Close"]).diff(1)[1:]
        return result

    def make_norm_diff(self):
        diff = self.make_diff()
        minn = min(diff)

        to_norm = diff - minn + 1
        norm, _ = stats.boxcox(to_norm)

        mean = np.mean(norm)
        result = norm - mean

        return result

    @staticmethod
    def analiza_statystyczna_szeregu(szereg_pandas: pd.Series, max_lag:int = 30, co_sprawdzamy:str="DANE PODSTAWOWE", wykres:bool=True, crit:str="AIC"):
        print("analiza_statystyczna_szeregu")

        if co_sprawdzamy is None:
            co_sprawdzamy = 'DANE NORMALNE'

        def stacjonarny(trend='ct', normalne_czy_nie=co_sprawdzamy):
            if trend == 'c':
                trend_display = "Constant only"
            elif trend == 'ct':
                trend_display = "Constant and trend"
            elif trend == "ctt":
                trend_display = "Constant, linear and quadriatic trend"
            elif trend == 'n':
                trend_display = "No trend"

            try:
                adf = adfuller(szereg_pandas, autolag=crit, regression=trend)
            except (TypeError, FloatingPointError, ValueError):
                adf = ["ERROR", "ERROR", "ERROR", "ERROR", {"5%": "ERROR"}]

            statistic = adf[0]
            pvalue = adf[1]
            crit_5 = adf[4]['5%']
            if pvalue != "ERROR":
                if pvalue < 0.05:
                    adf_res = "STACJONARNY"
                else:
                    adf_res = "NIESTACJONARNY"
            print(
                "---------------------------------------------------------------------------------------------------------------------------------------------------")
            print(
                "---------------------------------------------------------------------------------------------------------------------------------------------------")
            print(
                "--                                                            SPRAWDZENIE STACJONARNOŚCI                                                         --")
            print(
                f"--                                                                  {normalne_czy_nie}                                                                --")
            print(
                "---------------------------------------------------------------------------------------------------------------------------------------------------")
            print(
                "---------------------------------------------------------------------------------------------------------------------------------------------------")
            try:
                print(
                    f"|   {trend_display}   |   Statystyka ADF: {round(statistic, 3)}  |  Wartość krytyczna dla 5%: {round(crit_5, 3)}  |  pvalue: {round(pvalue, 3)}   |  {adf_res}  |")
            except TypeError:
                print(
                    f"|   {trend_display}   |   Statystyka ADF: ERROR  |  Wartość krytyczna dla 5%: ERROR  |  pvalue: ERROR   |  ERROR  |")

            print(
                "---------------------------------------------------------------------------------------------------------------------------------------------------")
            try:
                kpss_test = kpss(szereg_pandas, regression=trend)
            except (ValueError, FloatingPointError):
                kpss_test = ["ERROR", "ERROR", "ERROR", {"5%": "ERROR"}]
            statistic_kpss = kpss_test[0]
            pvalue_kpss = kpss_test[1]
            crit_5_kpss = kpss_test[3]['5%']
            if pvalue_kpss != "ERROR":
                if pvalue_kpss > 0.05:
                    kpss_res = "STACJONARNY"
                else:
                    kpss_res = "NIESTACJONARNY"
            else:
                kpss_res = "ERROR"

            if kpss_res == "ERROR":
                ogolny = "ERROR"
            elif adf_res == kpss_res == "STACJONARNY":
                ogolny = adf_res
            else:
                ogolny = "NIESTACJONARNY"

            print(
                "---------------------------------------------------------------------------------------------------------------------------------------------------")
            try:
                print(
                    f"|   {trend_display}   |   Statystyka KPSS: {round(statistic_kpss, 3)}  |  Wartość krytyczna dla 5%: {round(crit_5_kpss, 3)}  |  pvalue: {round(pvalue_kpss, 3)}   |  {kpss_res}  |")
            except TypeError:
                print(
                    f"|   {trend_display}   |   Statystyka KPSS: ERROR  |  Wartość krytyczna dla 5%: ERROR  |  pvalue: ERROR   |  ERROR  |")

            print(
                "---------------------------------------------------------------------------------------------------------------------------------------------------")
            print(
                "---------------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"                                                               REZULAT: {ogolny}")
            print(
                "---------------------------------------------------------------------------------------------------------------------------------------------------")

            #try:
            #    print(round(acorr_ljungbox(szereg_pandas, lags=10, return_df=True), 10))
            #    H, c, data = compute_Hc(szereg_pandas, kind='price', simplified=False)
            #except (FloatingPointError, ValueError):
            #    H = f"ERROR"
#
            #print("HURST: ", H)

        BDS = bds(szereg_pandas.values, max_dim=max_lag)[1]

        def acf_pacf(acf_lag=30, pacf_lag=30):
            if len(szereg_pandas) < (acf_lag and pacf_lag):
                acf_lag, pacf_lag = (len(szereg_pandas) / 2)-1, (len(szereg_pandas) / 2)-1

            this_pacf = pacf(szereg_pandas, nlags=pacf_lag)
            this_acf = acf(szereg_pandas, nlags=acf_lag)

            fig, axs = plt.subplots(4, figsize=(20, 10))
            if co_sprawdzamy is None:
                fig.suptitle('Testy autokorelacji')
            else:
                fig.suptitle(f'Testy autokorelacji dla {co_sprawdzamy}')

            axs[0].grid()
            axs[1].grid()
            axs[2].grid()

            axs[0].title.set_text("Test PACF")
            axs[0].plot(this_pacf, marker='o')
            axs[0].axhline(y=0, linestyle='--', color='gray')
            axs[0].axvline(x=0, linestyle='--', color='gray')
            axs[0].axhline(y=-1.96 / np.sqrt(len(szereg_pandas)), linestyle='--', color='gray')
            axs[0].axhline(y=1.96 / np.sqrt(len(szereg_pandas)), linestyle='--', color='gray')

            axs[1].title.set_text("Test ACF")
            axs[1].plot(this_acf, marker='o')
            axs[1].axhline(y=0, linestyle='--', color='gray')
            axs[1].axvline(x=0, linestyle='--', color='gray')
            axs[1].axhline(y=-1.96 / np.sqrt(len(szereg_pandas)), linestyle='--', color='gray')
            axs[1].axhline(y=1.96 / np.sqrt(len(szereg_pandas)), linestyle='--', color='gray')

            try:
                axs[2].plot(acorr_ljungbox(szereg_pandas, lags=max_lag, return_df=True)['lb_pvalue'])
                axs[2].title.set_text(f"Ljung-box pvalues dla {co_sprawdzamy}")
                axs[2].axhline(y=0.05, linestyle='--', color='gray')
                if acorr_ljungbox(szereg_pandas, lags=max_lag, return_df=True)['lb_pvalue'].all() < 0.05:
                    pass#axs[2].ylim(-0.01, 0.06)
                else:
                    pass#axs[2].ylim(-0.01, 1.01)
            except (FloatingPointError, ValueError):
                print("Ljung-Box nie mógł zostać wygenerowany - FloatingPointError")
            axs[3].title.set_text("Test BDS")
            axs[3].plot(BDS, marker='o')
            axs[3].axhline(y=0, linestyle='--', color='gray')
            axs[3].axvline(x=0, linestyle='--', color='gray')
            axs[3].axhline(y=0.05, linestyle='--', color='gray')
            axs[3].grid()
            plt.show()


        stacjonarny(trend='c')
        if wykres:
            acf_pacf(acf_lag=max_lag, pacf_lag=max_lag)

        print("Pvalue testu Jarque-Bera: ", scipy.stats.jarque_bera(szereg_pandas)[1])  # czy test jarque bera będzie istotny na małej próbie jak taka?
        print("Statystyka testu Jarque-Bera: ", scipy.stats.jarque_bera(szereg_pandas)[0])

        szereg_pandas.hist(bins=int(len(szereg_pandas)/3))
        plt.show()



getter = Get_Data(nazwa_instrumentu="^IXIC", start="2021-01-01", interval="1d")
