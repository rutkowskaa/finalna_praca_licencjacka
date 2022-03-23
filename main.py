import resources.Get_Data as Get_Data
from resources.single_data.RF_AR import RF_AR
import matplotlib.pyplot as plt
import pandas as pd


getter = Get_Data.Get_Data("^IXIC", "2022-02-15", "1h")#.make_norm_diff()
#print(getter)
#szereg = Get_Data.Get_Data("^IXIC", start="2021-09-20", end='2022-02-20', interval="1d").make_diff()
szereg = pd.read_csv("sim.csv")['x'][:100]
#print(szereg)
#getter.analiza_statystyczna_szeregu(szereg_pandas=szereg)

cart_ar = RF_AR(data=szereg, params={"lags": 1}, test_ratio=0.9)

opt = cart_ar.cross_validation_rolling_window(dlugosc_okna=1/2, params={
    "max_depth": 5,
    "max_n_estimators": 8,
    "min_sample_split": 5,
    "min_samples_leaf": 5
})

cart_ar.fit(params_fit={"max_depth": opt["depth"],
                        "n_estimators": opt["n_estimators"],
                        "min_sample_split": opt["min_sample_split"],
                        "min_samples_leaf": opt["min_samples_leaf"]})


forecasts = cart_ar.forecast_raw()

plt.plot(cart_ar.data_test.values[:50])
plt.plot(forecasts[:50], c='r')
plt.figure(figsize=(20,10))
plt.show()

#getter.analiza_statystyczna_szeregu(cart_ar.errors, co_sprawdzamy="reszty")
