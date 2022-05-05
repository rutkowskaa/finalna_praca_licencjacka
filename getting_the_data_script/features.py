import scipy
import yfinance as yf
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, pacf, acf, kpss
#import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import bds
import rpy2.robjects.packages as rpackages
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2

from sklearn.feature_selection import mutual_info_regression


def get_statistics(szereg_pandas: pd.Series, trend:str="n", max_lag:int = 30, co_sprawdzamy:str="DANE PODSTAWOWE", wykres:bool=True, crit:str="AIC"):
    #print("--------------------------------------------------------")


    #BDS = np.array([i for i in bds(szereg_pandas.values, max_dim=max_lag)[1]])

    ljung = np.array([i for i in acorr_ljungbox(szereg_pandas, lags=max_lag, return_df=True)['lb_pvalue']])

    ljung_sq = np.array([i for i in acorr_ljungbox(szereg_pandas ** 2, lags=max_lag, return_df=True)['lb_pvalue']])

    BDSs = bds(szereg_pandas, max_dim=max_lag+1)[1]

    mi = MI(szereg_pandas, max_lag)

    return ljung, ljung_sq, BDSs, mi

def gph():
    frac = rpackages.importr('fracdiff')

    a = robjects.r(f'''
        fdGPH({[1,2,3]})
    ''')


def MI(series, lags):
    datami = pd.DataFrame()
    for i in range(0, lags):
        datami.loc[:, f"{i}"] = series.shift(i)

    result = np.array([i for i in mutual_info_regression(y=series[lags:], X=datami[lags:])])
    return result

def BDS(series, lags):
    datami = pd.DataFrame()
    for i in range(0, lags):
        datami.loc[:, f"{i}"] = series.shift(i)
    result = np.array([i for i in bds(series, max_dim=lags)[1]])
    return result