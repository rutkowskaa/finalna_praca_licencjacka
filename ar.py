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

#  Pomijam analizę statystyczną
szereg_norm = Get_Data.Get_Data("^IXIC", start="2018-08-22", end="2019-10-30",
                                interval="1d").make_diff()

rf_ar = CART_AR(data=szereg_norm, params={"lags": 1}, test_ratio=0.7)

#Uwaga! Przy zmienianiu algorytmu należy pilnować, aby w 'params' było dokładnie tyle parametrów ile dany model przyjmuje. Inaczej może wyskoczyć błąd.
# Zauważyłem, że zazwyczaj mniejsze okno = lepsza prognoza. Nie wiem dlaczego, ale warto mieć na uwadze.

opt = rf_ar.cross_validation_rolling_window_julia(dlugosc_okna=1 / 2, params={
    "max_depth": 4,
    #"n_estimators": 5,
    "min_samples_split": 45,
    "min_samples_leaf": 25
})

print("Optymalne ustawienia: ", opt)
rf_ar.fit(opt)


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

    #
    # Próbowałem wyprowadzić MSE penalizujące przeciwny kierunek prognozy do wartości rzeczywistej,
    # ale nie zauważyłem żadnej różnicy
    #
    print("SUMA: ", sum(np.squeeze(data_test.values) * np.array(result) > 0),
          " / ", len(result))

show_forcasts(rf_ar.forecast_raw(), rf_ar.data, rf_ar.data_test)
