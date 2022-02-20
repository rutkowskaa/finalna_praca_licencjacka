from resources.CART_AR import CART_AR as CART_AR
import resources.Get_Data as Get_Data
import resources.KNN_AR as KNN_AR
import resources.RF_AR as RF_AR
import matplotlib.pyplot as plt

getter = Get_Data.Get_Data("BTC-USD", "2022-02-15", "1m")#.make_norm_diff()


getter.analiza_statystyczna_szeregu(szereg_pandas=getter.make_log_diff())

cart_ar = CART_AR(data=getter.make_log_diff(), params={"lags": 5}, test_ratio=0.95)
opt = cart_ar.cross_validation_rolling_window(dlugosc_okna=1/3, max_depth=20)

cart_ar.fit(params_fit={"max_depth": opt})

forecasts = cart_ar.forecast_raw()

plt.plot(cart_ar.data_test.values)
plt.plot(forecasts, c='r')
plt.figure(figsize=(20,10))
plt.show()
getter.analiza_statystyczna_szeregu(cart_ar.errors, co_sprawdzamy="reszty")