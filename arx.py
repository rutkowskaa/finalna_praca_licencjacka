import numpy as np
import resources.Get_Data as Get_Data
from resources.single_data.RF_AR import RF_AR
from resources.single_data.CART_AR import CART_AR
from resources.single_data.KNN_AR import KNN_AR
from resources.vectorised_data.MISO.ARX_repr.KNN_ARX import KNN_ARX
from resources.vectorised_data.MISO.ARX_repr.CART_ARX import CART_ARX
from resources.vectorised_data.MISO.ARX_repr.RF_ARX import RF_ARX
import matplotlib.pyplot as plt
import pandas as pd
import julia
from julia import Pkg
from julia import Main
import Get_Vectorised_Data
Pkg.add("DecisionTree")
Main.using("DecisionTree")


szereg_norm = Get_Vectorised_Data.Get_Vectorised_Data(["^IXIC", "^GSPC"], start="2020-09-20", end='2022-01-20', interval="1d").make_diff()

cart_arx = CART_ARX(data=szereg_norm, params={"lags": 1}, test_ratio=0.7, to_predict="^IXIC")

opt = cart_arx.cross_validation_rolling_window_julia(dlugosc_okna=1/2, params={
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 10
})
print("OPT: ", opt)
cart_arx.fit(opt)
plt.plot(np.cumsum(cart_arx.data_test.values))
plt.plot(np.cumsum(cart_arx.forecast_raw()), c='r')
plt.show()