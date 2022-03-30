import Get_Data
import pandas as pd
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")


class KNN_AR():
    def __init__(self, data: pd.Series = None, params: dict = None, plot_predicted_insample: bool = False,
                 test_ratio: float = 0.95):
        self.params = params
        self.test_ratio = test_ratio
        self.params = params
        self.lags = params["lags"]

        self.all_data = data.iloc[self.lags:]
        self.all_data_zapas = data

        self.data = data
        self.data1 = data[self.lags:]
        self.data = data[:int(test_ratio * len(self.all_data))]
        self.data_test = self.data1[int(test_ratio * len(self.all_data)):]

        X = self.make_lags(self.all_data_zapas, params["lags"])
        self.all_Xs = X

        self.X = X.iloc[:int(test_ratio * len(self.all_data))]
        self.X_test = X.iloc[int(test_ratio * len(self.all_data)):]

    def make_lags(self, input: pd.DataFrame, lags):
        output = pd.DataFrame()
        for i in range(1, lags + 1):
            output.insert(loc=len(output.columns), column=f"{i}", value=input.shift(i))
        return output[lags:]

    def fit(self, params_fit):

        self.model = neighbors.KNeighborsRegressor(n_neighbors=int(params_fit["k"]),
                                                   weights=params_fit["weights"],
                                                   algorithm="brute",
                                                   p=int(params_fit["p"]))
        print(self.X, self.data)
        self.model.fit(X=self.X, y=self.data)
        self.params = params_fit
        print("fit")

    def cross_validation_rolling_window(self, dlugosc_okna: int, params:dict):
        """
        :param dlugosc_okna: długość okna branego pod uwagę do trenowania modelu. To powinien być ułamek.
        :param k_max: Maksymalna wartość parametru k brana pod uwagę
        :return:
        """
        # dlugosc_okna = 1-dlugosc_okna
        self.dlugosc_okna = dlugosc_okna

        def MSE_cross_val(preds):
            actual = self.data[self.prog:]
            mse = (1 / len(preds)) * sum((actual - preds) ** 2)
            return mse

        i = 0  # debug, nie usuwac
        all_preds = np.array([])
        pure_errors = np.array([])
        self.prog = int(dlugosc_okna * len(self.data))


        for k in range(1, params["k_max"]):
            print(k)
            for weight in params["weights"]:
                for p in params["p"]:
                    pred = np.array([])
                    for i in range(self.prog, len(self.data)):

                        train_x = self.X.iloc[:i, :]
                        train_y = self.data.iloc[:i]

                        valid = neighbors.KNeighborsRegressor(n_neighbors=k, weights=weight, p=p)
                        valid.fit(X=train_x, y=train_y)

                        lim = self.X.iloc[i, :]
                        pred = np.append(pred, valid.predict(X=np.array([lim.values])))


                    all_preds = np.append(all_preds, [k, weight, p, pred])
                    pure_errors = np.append(pure_errors, [k, weight, p, MSE_cross_val(preds=pred)])  # RMSE RMSE_cross_val(preds=pred)]
                    pure_errors = pure_errors.reshape(-1, len(params) + 1)

        bledy = np.array(pure_errors[:, len(params)])
        min_errors = min(bledy)
        optimal_params = np.where(bledy == min_errors)[0]#[0] + 1
        result = pure_errors[optimal_params, :][0][:len(params)]

        to_ret = {
            "k_max": result[0],
            "weights": result[1],
            "p": result[2]
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
            model = neighbors.KNeighborsRegressor(n_neighbors=int(self.params["k"]),
                                                       weights=self.params["weights"],
                                                       algorithm="brute",
                                                       p=int(self.params["p"]))
            model.fit(X=to_test_x, y=to_test_y)
            forecasts = np.append(forecasts, model.predict(np.array([self.all_Xs.iloc[i]])))

        self.errors = self.data_test - forecasts

        print("forecast_raw")
        return forecasts

    def forecast(self):
        print("forecast")

    def analizuj_reszty(self):
        print("analizuj_reszty")

# getter = Get_Data.Get_Data("AAPL", "2022-01-01", "1h")#.make_diff()
#
# getter.analiza_statystyczna_szeregu(szereg_pandas=getter.make_diff())
#
# knn_ar = KNN_AR(data=getter.make_diff(), params={"lags": 3})
# opt = knn_ar.cross_validation_rolling_window(dlugosc_okna=1/3, k_max=15)
#
# knn_ar.fit(params_fit={"k": opt})
#
# forecasts = knn_ar.forecast_raw()
#
# plt.plot(knn_ar.data_test.values)
# plt.plot(forecasts, c='r')
# plt.figure(figsize=(20,10))
# plt.show()
#
# getter.analiza_statystyczna_szeregu(knn_ar.errors, co_sprawdzamy="reszty")
