import Get_Data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import Get_Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import julia
from julia import Main
import warnings

warnings.filterwarnings("ignore")


class CART_AR():
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
        print(params_fit)
        self.model = DecisionTreeRegressor(max_depth=params_fit["max_depth"],
                                           min_samples_split=params_fit["min_sample_split"],
                                           min_samples_leaf=params_fit["min_samples_leaf"])
        self.model.fit(X=self.X, y=self.data)
        self.params = params_fit
        print("fit")

    def cross_validation_rolling_window(self, dlugosc_okna: int, params: dict, verbose=True):
        """
        :param dlugosc_okna: długość okna branego pod uwagę do trenowania modelu. To powinien być ułamek.
        :param max_depth: Maksymalna wartość parametru k brana pod uwagę
        :return:
        """
        self.dlugosc_okna = dlugosc_okna

        # Tutaj zdefiniowane są funkcje błędów
        def MSE_cross_val(preds, prog):
            actual = self.data[self.prog:]
            mse = (1 / len(preds)) * sum((actual - preds) ** 2)
            return mse

        all_preds = np.array([])
        pure_errors = np.array([])
        self.prog = int(dlugosc_okna * len(self.data))

        for depth in range(1, params["max_depth"]):
            for sample in range(2, params["min_sample_split"]):
                for leaf in range(2, params["min_samples_leaf"]):

                    print(depth)
                    pred = np.array([])

                    for i in range(self.prog, len(self.data)):
                        print("TU ", i - self.prog, i)
                        train_x = self.X.iloc[i - self.prog: i]
                        train_y = self.data.iloc[i - self.prog: i]
                        print(len(train_x))
                        valid = DecisionTreeRegressor(max_depth=depth,
                                                      min_samples_split=sample,
                                                      min_samples_leaf=leaf)
                        valid.fit(X=train_x, y=train_y)

                        lim = self.X.iloc[i, :]
                        pred = np.append(pred, valid.predict(X=[lim.values]))

                    all_preds = np.append(all_preds, [depth, pred])
                    pure_errors = np.append(pure_errors, [int(depth), int(sample), int(leaf),
                                                          MSE_cross_val(preds=pred, prog=self.prog)])
                    pure_errors = pure_errors.reshape(-1, len(params) + 1)

        bledy = np.array(pure_errors[:, len(params)])

        min_errors = min(bledy)
        opt_depth = np.where(bledy == min_errors)[0]
        result = pure_errors[opt_depth][0][0:len(params)]
        to_ret = {
            "depth": result[0],
            "min_sample_split": result[1],
            "min_samples_leaf": result[2]
        }

        return to_ret

    def cross_validation_rolling_window_julia(self, dlugosc_okna: int, params: dict, verbose=True):
        """
        :param dlugosc_okna: długość okna branego pod uwagę do trenowania modelu. To powinien być ułamek.
        :param max_depth: Maksymalna wartość parametru k brana pod uwagę
        :return:
        """
        self.prog = int(dlugosc_okna * len(self.data))
        j = julia.Julia()
        julia.install()
        # Main.using("DecisionTree")

        # Main.using("RandomForest")

        Main.dict = {"dlugosc_okna": dlugosc_okna,
                     "prog": self.prog,
                     "data": self.data.values,
                     "X": self.X.values,
                     "params": params}
        Main.include('resources/fast_jl/rf_cross_val.jl')
        # Main.include('RandomForest.jl')
        fn = Main.rf_cross_val(Main.dict)
        return fn

    def predict(self):
        self.predictions = self.model.predict(self.X)
        return self.predictions

    def forecast_raw(self):
        forecasts = np.array([])

        for i in range(len(self.data), len(self.all_data)):
            to_test_x = self.all_Xs[i - self.prog: i]
            to_test_y = self.all_data[i - self.prog: i]

            model = DecisionTreeRegressor(max_depth=self.params["max_depth"])
            model.fit(X=to_test_x, y=to_test_y)
            forecasts = np.append(forecasts, model.predict([self.all_Xs.iloc[i]]))

        self.errors = self.data_test - forecasts

        print("forecast_raw")
        return forecasts

    def forecast(self):
        print("forecast")

    def analizuj_reszty(self):
        print("analizuj_reszty")
