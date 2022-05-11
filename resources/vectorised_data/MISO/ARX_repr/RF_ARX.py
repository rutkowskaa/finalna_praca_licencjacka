import Get_Data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import Get_Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import julia
from julia import Main
from julia import Pkg
import warnings
Pkg.add("DecisionTree")
Main.using("DecisionTree")
warnings.filterwarnings("ignore")


class RF_ARX():
    def __init__(self, data: pd.Series = None, to_predict: str = None, params: dict = None,
                 plot_predicted_insample: bool = False, test_ratio: float = 0.95, dlugosc_okna: float = 1/3):

        self.predictions = None
        if to_predict == None:
            self.to_predict = data.columns[0]
            print(f"NIE PODANO ZMIENNEJ OBJAŚNIANEJ, WYBRANO AUTOMATYCZNIE: {data.columns[0]}")
        else:
            self.to_predict = to_predict

        ###########################################
        ###########################################
        #  przypisanie danych do obiektu

        self.dlugosc_okna = dlugosc_okna
        self.params = params
        self.test_ratio = test_ratio
        self.params = params
        self.lags = params["lags"]
        self.prog = int(dlugosc_okna * len(data))
        self.ratio_int = int(len(data) * self.test_ratio)
        ###########################################
        ###########################################
        #  przetwarzanie danych Y i X
        #  zmienna self.data = Y
        #  zmienna self.X = X

        if data is pd.Series:
            data = data.to_frame()
        self.all_data = data
        X = self.make_lags(self.all_data, params["lags"])

        self.all_data = self.all_data[self.lags:][self.to_predict]
        self.data = self.all_data[:self.ratio_int]
        self.data_test = self.all_data[self.ratio_int:]

        self.all_Xs = X
        self.X = X.iloc[:self.ratio_int]
        self.X_test = X.iloc[self.ratio_int:]

    def make_lags(self, input: pd.DataFrame, lags):
        output = pd.DataFrame()

        for i in range(1, lags + 1):
            for col in input.columns:
                output[f"{col}{i}"] = input[col].shift(i)
        return output[lags:]

    def fit(self, params_fit):
        print(params_fit)
        self.model = RandomForestRegressor(max_depth=params_fit["max_depth"],
                                           n_estimators=int(params_fit["n_estimators"]),
                                           min_samples_split=int(params_fit["min_samples_split"]),
                                           min_samples_leaf=int(params_fit["min_samples_leaf"]),
                                           bootstrap=False)
        self.model.fit(X=self.X, y=self.data)
        self.params = params_fit
        print("fit")

    #def cross_validation_rolling_window(self, dlugosc_okna: int, params: dict, verbose=True):
    #    """
    #    :param dlugosc_okna: długość okna branego pod uwagę do trenowania modelu. To powinien być ułamek.
    #    :param max_depth: Maksymalna wartość parametru k brana pod uwagę
    #    :return:
    #    """
    #    self.dlugosc_okna = dlugosc_okna
#
    #    # Tutaj zdefiniowane są funkcje błędów
    #    def MSE_cross_val(preds, prog):
    #        actual = self.data[self.prog:][self.to_predict]
    #        mse = (1 / len(preds)) * sum((actual - preds) ** 2)
    #        return mse
#
    #    all_preds = np.array([])
    #    pure_errors = np.array([])
    #    self.prog = int(dlugosc_okna * len(self.data))
#
    #    for depth in range(1, params["max_depth"]):
    #        for sample in range(2, params["min_sample_split"]):
    #            for leaf in range(2, params["min_samples_leaf"]):
#
    #                print(depth, sample, leaf)
    #                pred = np.array([])
#
    #                for i in range(self.prog, len(self.data)):
    #                    #print("TU ", i - self.prog, i)
    #                    train_x = self.X.iloc[i - self.prog: i]
    #                    train_y = self.data.iloc[i - self.prog: i][self.to_predict]
    #                    valid = RandomForestRegressor(max_depth=depth,
    #                                                  min_samples_split=sample,
    #                                                  min_samples_leaf=leaf)
    #                    valid.fit(X=train_x, y=train_y)
#
    #                    lim = self.X.iloc[i, :]
    #                    pred = np.append(pred, valid.predict(X=[lim.values]))
#
    #                all_preds = np.append(all_preds, [depth, pred])
    #                pure_errors = np.append(pure_errors, [int(depth), int(sample), int(leaf),
    #                                                      MSE_cross_val(preds=pred, prog=self.prog)])
    #                pure_errors = pure_errors.reshape(-1, len(params))
#
    #    print(pure_errors)
    #    bledy = np.array(pure_errors[:, len(params) - 1])
#
    #    min_errors = min(bledy)
    #    opt_depth = np.where(bledy == min_errors)[0]
    #    result = pure_errors[opt_depth][0][0:len(params)]
    #    to_ret = {
    #        "max_depth": int(result[0]),
    #        "n_estimators": int(result[1]),
    #        "min_sample_split": int(result[2]),
    #        "min_samples_leaf": int(result[3])
    #    }
#
    #    return to_ret

    def cross_validation_rolling_window_julia(self, dlugosc_okna: float, params: dict, verbose=True):

        self.prog = int(dlugosc_okna * len(self.data))

        Main.dict = {"dlugosc_okna": dlugosc_okna,
                     "prog": self.prog,
                     "data": self.data.values,
                     "X": self.X.values,
                     "params": params}

        Main.include('resources/fast_jl/vector_rf_cross_val.jl')
        fn = Main.vector_rf_cross_val(Main.dict)
        return fn

    def predict(self):
        self.predictions = self.model.predict(self.X)
        return self.predictions

    def forecast_raw(self):
        forecasts = np.array([])
        model = RandomForestRegressor(max_depth=self.params["max_depth"],
                                      n_estimators=self.params["n_estimators"],
                                      min_samples_split=self.params["min_samples_split"],
                                      min_samples_leaf=self.params["min_samples_leaf"],
                                      bootstrap=False)

        for i in range(len(self.data), len(self.all_data)):
            to_test_x = self.all_Xs[i - self.prog: i]
            to_test_y = self.all_data[i - self.prog: i]

            model.fit(X=to_test_x, y=to_test_y)
            forecasts = np.append(forecasts, model.predict([self.all_Xs.iloc[i, :]]))

        self.errors = self.data_test - forecasts

        print("forecast_raw")
        return forecasts

    def forecast(self):
        print("forecast")

    def analizuj_reszty(self):
        print("analizuj_reszty")
