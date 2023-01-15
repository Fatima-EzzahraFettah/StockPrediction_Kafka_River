import yfinance as yf
from datetime import date
from datetime import timedelta
from river import metrics
from river.ensemble import AdaptiveRandomForestRegressor
from river.tree import HoeffdingAdaptiveTreeRegressor
from river.linear_model import LinearRegression
from river.stream import iter_pandas
from river.preprocessing import  StandardScaler
from river import time_series, preprocessing, optim
import numpy as np
import random


def print_progress(sample_id,acc):
    print(f'Samples processed: {sample_id}')
    print(acc)

class TrainModelOffline:
    def __init__(self, company_name, periods=4):
        self.model_lr = (StandardScaler() | LinearRegression())
        self.model_rf = (StandardScaler() | HoeffdingAdaptiveTreeRegressor(seed=42))
        self.model_ar = (time_series.SNARIMAX(
            p=periods,
            d=0,
            q=0,
            m=periods,
            regressor=(preprocessing.StandardScaler() | LinearRegression(intercept_init=110,
                                                                         optimizer=optim.SGD(0.01),
                                                                         intercept_lr=0.3)
                       )
        )
        )

        self.model_ma = (time_series.SNARIMAX(
            p=0,
            d=0,
            q=periods,
            m=periods,
            regressor=(preprocessing.StandardScaler() | LinearRegression(intercept_init=110,
                                                                         optimizer=optim.SGD(0.01),
                                                                         intercept_lr=0.3)
                       )
        )
        )
        self.model_arma = (time_series.SNARIMAX(
            p=periods,
            d=0,
            q=periods,
            m=periods,
            regressor=(preprocessing.StandardScaler() | LinearRegression(intercept_init=110,
                                                                         optimizer=optim.SGD(0.01),
                                                                         intercept_lr=0.3)
                       )
        )
        )
        self.model_lstm = None
        self.company_name = company_name
        current_data = date.today()
        start_date = current_data - timedelta(days=1)
        self.data = yf.download(self.company_name, start=start_date, end = current_data ,interval='1m')
        self.data = np.log(self.data)
        self.X = self.data[['Open', 'High', 'Low']]
        self.X = self.X.shift(periods=1)
        self.Y = self.data['Close']

    def train_model_offline(self,model_name = "linear-regression", n_wait = 100,verbose = True, min_train_start = 10):
        print("***************Model Launched: ", model_name, "*********************")
        if model_name == "linear-regression":
            self.model = self.model_lr
        elif model_name =="Random-Forest":
            self.model = self.model_rf
        elif model_name == "Auto-Regressive":
            self.model = self.model_ar
        elif model_name == "Moving-Average":
            self.model = self.model_ma
        elif model_name == "ARMA":
            self.model = self.model_arma
        elif model_name == "LSTM":
            self.model = self.model_lstm
        else:
            self.model = None

        acc = metrics.MAE()
        raw_results = []
        stream = iter_pandas(X=self.X, y=self.Y)
        price_predictions = []
        for i, (x, y) in enumerate(stream):
            if i <= min_train_start :
                y_pred = random.uniform(58,59)
            else:
                if model_name in ["Moving-Average", "Auto-Regressive", "ARMA"]:
                    y_pred = self.model.forecast(horizon=1)[0]
                else:
                    y_pred = self.model.predict_one(x)
            price_predictions.append(np.exp(y_pred))
            acc.update(y_true=y, y_pred=y_pred)
            if i % n_wait == 0 and i > 0:
                if verbose:
                    print_progress(i, acc)
                raw_results.append([model_name, i,np.exp(y_pred), np.exp(y), acc.get()])
            if i > 2:
                if model_name in ["linear-regression","Random-Forest"]:
                    self.model.learn_one(x, y)
                else:
                    self.model.learn_one(y)






