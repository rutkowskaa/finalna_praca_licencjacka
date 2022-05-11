import Get_Data
import pandas as pd
from sklearn import neighbors
import Get_Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")


class KNN_ARX():
    def __init__(self, data: pd.Series = None, to_predict: str = None, params: dict = None, plot_predicted_insample: bool = False, test_ratio: float = 0.95, dlugosc_okna: float = 1/3):
        self.reshape_adjustment_param = len(data.columns)
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
        self.params = params_fit
        print(params_fit)
        predictions = np.array([])
        model = neighbors.KNeighborsRegressor(n_neighbors=params_fit["k_neighbors"], weights=params_fit["weights"],
                                              p=params_fit["p"])

        for i in range(self.prog, len(self.data)):

            to_test_x = self.X[i - self.prog: i]
            to_test_y = self.data[i - self.prog: i]
            #print(to_test_y.values.shape, to_test_x.values.shape)

            model.fit(X=to_test_x, y=to_test_y)
            predictions = np.append(predictions, model.predict(self.X.iloc[i].values.reshape(-1, self.lags * self.reshape_adjustment_param)))

        self.model = model.fit(X=self.X[len(self.data) - self.prog: len(self.data)], y=self.data[len(self.data) - self.prog: len(self.data)])
        self.predictions = predictions
        self.errors = self.data[self.prog:].values - self.predictions

        print("fit")

    def cross_validation_rolling_window(self, dlugosc_okna: float, params: dict):
        """
        :param dlugosc_okna: długość okna branego pod uwagę do trenowania modelu. To powinien być ułamek.
        :param k_max: Maksymalna wartość parametru k brana pod uwagę
        :return:
        """
        # dlugosc_okna = 1-dlugosc_okna
        self.dlugosc_okna = dlugosc_okna

        def MSE(actual, preds):
            mse = (1 / len(preds)) * ((actual - preds) ** 2).sum()
            return mse

        i = 0  # debug, nie usuwac
        all_preds = np.array([])
        pure_errors = np.array([])
        self.prog = int(dlugosc_okna * len(self.data))

        for k in range(1, params["k_neighbors"]):
            print(k)
            for weight in params["weights"]:
                for p in params["p"]:
                    pred = np.array([])
                    for i in range(self.prog, len(self.data)):
                        train_x = self.X.iloc[i - self.prog: i]
                        train_y = self.data.iloc[i - self.prog: i]
                        #print(i - self.prog, i)

                        valid = neighbors.KNeighborsRegressor(n_neighbors=k, weights=weight, p=p)
                        valid.fit(X=train_x, y=train_y)

                        lim = self.X.iloc[i, :]
                        pred = np.append(pred, valid.predict(X=np.array([lim.values])))

                    all_preds = np.append(all_preds, [k, weight, p, pred])
                    pure_errors = np.append(pure_errors, [k, weight, p, MSE(self.data.values[self.prog:], pred)])
        pure_errors = pure_errors.reshape(-1, 4)
        only_errors = pure_errors[:, len(params)].astype(np.float64)
        min_error = min(only_errors)

        where_are_the_params = np.where(only_errors == min_error)[0][0]
        best_params = pure_errors[where_are_the_params, :]
        to_ret = {
            "k_neighbors": int(best_params[0]),
            "weights": best_params[1],
            "p": int(best_params[2])
        }

        print("cross_validation_rolling_window")
        return to_ret

    def predict(self):
        self.predictions = self.model.predict(self.X)
        return self.predictions

    def forecast_raw(self):
        forecasts = np.array([])
        for i in range(len(self.data), len(self.all_data)):
            to_test_x = self.all_Xs[i - self.prog: i]
            to_test_y = self.all_data[i - self.prog: i]

            model = neighbors.KNeighborsRegressor(n_neighbors=int(self.params["k_neighbors"]),
                                                  weights=self.params["weights"],
                                                  algorithm="brute",
                                                  p=int(self.params["p"]))

            model.fit(X=to_test_x, y=to_test_y)
            forecasts = np.append(forecasts, model.predict(np.array([self.all_Xs.iloc[i, :]])))

        self.errors = self.data_test - forecasts

        print("forecast_raw")
        return forecasts

    def forecast(self):
        print("forecast")

    def analizuj_reszty(self):
        print("analizuj_reszty")
