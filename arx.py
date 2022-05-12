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
Pkg.add("DecisionTree")  # bez tych dwóch walidacja nie zadziała
Main.using("DecisionTree")  # bez tych dwóch walidacja nie zadziała


# Pierwsza opcja (do odkomentowania) zawiera macierz X jako [Nasdaq, Bitcoin-dollar, Dow-Jones, SP500]
# Druga opcja zawiera zwroty z Nasdaq oraz zwroty wolumenu z średnią równą 0
# ----------------------------------------------------------------------------------------------------------
# Opcja nr. 1
#szereg_norm = Get_Vectorised_Data.Get_Vectorised_Data(["^IXIC", "BTC-USD", "^DJI", "^GSPC"], start="2018-08-22", end="2019-10-30",
#                                                      interval="1d").make_diff()
#-------------------------------------------------------------------------------------------------------------
#Opcja nr. 2
szereg_norm = Get_Data.Get_Data("^IXIC", start="2018-08-22", end="2019-10-30", interval="1d").make_diff_with_volume()
szereg_norm["Volume"] = szereg_norm["Volume"] - szereg_norm["Volume"].mean()
#________________________________________________________________________________________________________


cart_arx = RF_ARX(data=szereg_norm, params={"lags": 1}, test_ratio=0.9, to_predict="Close")

#Uwaga! Przy zmienianiu algorytmu należy pilnować, aby w 'params' było dokładnie tyle parametrów ile dany model przyjmuje. Inaczej może wyskoczyć błąd.
opt = cart_arx.cross_validation_rolling_window_julia(dlugosc_okna=1/20, params={
    "max_depth": 5,
    "n_estimators": 5,
    "min_samples_split": 5,
    "min_samples_leaf": 5
})
print("Optymalne ustawienia: ", opt)
cart_arx.fit(opt)

def show_forcasts(forecasts, data, data_test):

    last = data.iloc[-1]
    result = np.array([last + forecasts[0]])
    cum_data_test = np.cumsum(data_test).values

    for i in range(1, len(forecasts)):
        result = np.append(result, cum_data_test[i - 1] + forecasts[i])

    plt.plot(cum_data_test)
    plt.plot(result, c='r')
    plt.grid()
    plt.show()

    print("SUMA: ", sum(np.squeeze(data_test.values) * np.array(result) > 0),
          " / ", len(result))


show_forcasts(cart_arx.forecast_raw(), cart_arx.data, cart_arx.data_test)