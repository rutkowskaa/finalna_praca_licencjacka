import resources.Get_Vectorised_Data as Get_Vectorised_Data
from vectorised_data.MIMO.CART_ARX import CART_ARX

szereg = Get_Vectorised_Data.Get_Vectorised_Data(["^IXIC", "^GSPC"], start="2021-09-20", end='2022-02-20', interval="1d").make_diff()
#szereg = pd.read_csv("sim.csv")['x'][:300]
#getter.analiza_statystyczna_szeregu(szereg_pandas=szereg)

cart_ar = CART_ARX(data=szereg, params={"lags": 0}, test_ratio=0.9)
opt = cart_ar.cross_validation_rolling_window(dlugosc_okna=1/5, max_depth=10)

#cart_ar.fit(params_fit={"max_depth": opt})