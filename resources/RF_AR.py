import Get_Data

class KNN_AR():
    def __init__(self, data:Get_Data=None, brute_data=None, params:dict()=None, plot_predicted_insample:bool=False):
        print("Konstruktor")
        self.fit()
        self.cross_validation_rolling_window()
        self.predict()
        self.forecast_raw()
        self.forecast()
        self.analizuj_reszty()

    def fit(self, params:dict()):
        print("fit")

    def cross_validation_rolling_window(self, dlugosc_okna:int, lags:int=None):
        print("cross_validation_rolling_window")

    def predict(self):
        print("predict")

    def forecast_raw(self):
        print("forecast_raw")

    def forecast(self):
        print("forecast")

    def analizuj_reszty(self):
        print("analizuj_reszty")

