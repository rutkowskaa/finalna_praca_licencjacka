import Get_Data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import Get_Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class CART_ARX():
    def __init__(self, data:pd.Series=None, params:dict()=None, plot_predicted_insample:bool=False, test_ratio:float=0.95):

        self.params = params
        self.test_ratio = test_ratio
        self.params = params
        self.lags = params["lags"]


        self.all_data = data.iloc[self.lags:]
        self.all_data_zapas = data

        self.data = data
        self.data1 = data[self.lags:]
        self.data = data[:int(test_ratio*len(self.all_data))]
        self.data_test = self.data1[int(test_ratio*len(self.all_data)):]
        X = self.make_lags(self.all_data_zapas, params["lags"])
        self.all_Xs = X

        self.X = X.iloc[:int(test_ratio*len(self.all_data))]
        self.X_test = X.iloc[int(test_ratio*len(self.all_data)):]



    def make_lags(self, input:pd.DataFrame, lags):
        output = pd.DataFrame()
        for i in range(1, lags + 1):
            print(i)
            output.insert(loc=len(output.columns), column=[f"{i}", f"{i}"], value=input.shift(i))
        return output[lags:]



    def fit(self, params_fit):
        print(params_fit)
        self.model = DecisionTreeRegressor(max_depth=params_fit["max_depth"])
        self.model.fit(X=self.X, y=self.data)
        self.params = params_fit
        print("fit")

    def cross_validation_rolling_window(self, dlugosc_okna:int, max_depth:int=3):
        """
        :param dlugosc_okna: długość okna branego pod uwagę do trenowania modelu. To powinien być ułamek.
        :param max_depth: Maksymalna wartość parametru k brana pod uwagę
        :return:
        """
        self.dlugosc_okna = dlugosc_okna
        def MSE_cross_val(preds):
            actual = self.data[self.prog:]
            mse = (1 / len(preds)) * sum((actual - preds) ** 2)
            return mse
        all_preds = np.array([])
        pure_errors = np.array([])
        self.prog = int(dlugosc_okna*len(self.data))
        for depth in range(1, max_depth):
            print("Current depth: ", depth)
            pred = np.array([])

            for i in range(int(dlugosc_okna*len(self.data)), len(self.data)):

                train_x = self.X.iloc[:i]
                train_y = self.data.iloc[:i]
                print("TU", self.X.iloc)
                valid = DecisionTreeRegressor(max_depth=depth)
                valid.fit(X=train_x, y=train_y)

                lim = self.X.iloc[i, :]
                pred = np.append(pred, valid.predict(X=[lim.values]))


            all_preds = np.append(all_preds, [depth, pred])
            pure_errors = np.append(pure_errors, [depth, MSE_cross_val(preds=pred)]) # RMSE RMSE_cross_val(preds=pred)]
            pure_errors = pure_errors.reshape(-1, 2)

        bledy = np.array(pure_errors[:, 1])
        min_errors = min(bledy)
        opt_depth = np.where(bledy==min_errors)[0][0] + 1


        print("OPTYMALNA WARTOŚĆ PARAMETRU MAX_DEPTH: ", opt_depth)
        return opt_depth

    def predict(self):
        self.predictions = self.model.predict(self.X)
        return self.predictions

    def forecast_raw(self):
        forecasts = np.array([])

        for i in range(len(self.data), len(self.all_data)):
            to_test_x = self.all_Xs[i - self.prog : i]
            to_test_y = self.all_data[i - self.prog : i]

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


