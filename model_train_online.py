from model_train_offline import TrainModelOffline
from river import metrics
import matplotlib.pyplot as plt
from kafka_consumers import consumer_init
import random
import json
import numpy as np
from matplotlib.dates import DateFormatter

class TrainModelOnline:
    def __init__(self, company_name, topic_name, periods=4):
        self.models_dict = {"Linear Regression":'lr',"Random Forest":'rf',"Moving Average":'ma',"Auto Regressive":'ar',"ARMA":'arma'}

        self.models_list = list(self.models_dict.keys())
        self.consumer = consumer_init(topic_name)
        self.periods = periods
        self.company_name = company_name
        self.features = ['Open', 'High', 'Low']

        offline_train = TrainModelOffline(self.company_name)
        offline_train.train_model_offline(model_name="linear-regression")
        self.model_lr = offline_train.model

        offline_train.train_model_offline(model_name="Random-Forest")
        self.model_rf = offline_train.model

        offline_train.train_model_offline(model_name="Auto-Regressive")
        self.model_ar = offline_train.model

        offline_train.train_model_offline(model_name="Moving-Average")
        self.model_ma = offline_train.model

        offline_train.train_model_offline(model_name="ARMA")
        self.model_arma = offline_train.model


    def train_model_online(self):
        fig, axs = plt.subplots(2,5, figsize=(24, 15))
        fig.suptitle(self.company_name + " stock price trading")
        is_plotted = False

        while True:
            mae_score_lr = metrics.MAE()
            mae_score_rf = metrics.MAE()
            mae_score_ma = metrics.MAE()
            mae_score_ar = metrics.MAE()
            mae_score_arma = metrics.MAE()
            self.close_price_stream = []
            self.close_price_pred = dict.fromkeys(self.models_dict.keys(), [])
            self.scoresmae = dict.fromkeys(self.models_dict.keys(), [])
            i=0
            stream = []
            timestamps = []
            for record in self.consumer:
                record_dict = json.loads(record.value)
                y = record_dict['Close']
                self.close_price_stream.append(y)
                stream.append(record_dict)
                date_time_stamp = record_dict['Datetime']
                timestamps.append(date_time_stamp)
                if i < 2:
                    y_pred = random.uniform(58, 59)
                    for key, value in self.close_price_pred.items():
                        value_list = value.copy()
                        value_list.append(y_pred)
                        self.close_price_pred.update({key:value_list})

                else:
                    x = {k: np.log(v) for k, v in stream[i-1]. items() if k in self.features}

                    y_pred_lr = self.model_lr.predict_one(x)
                    y_pred_rf = self.model_rf.predict_one(x)
                    y_pred_ar = self.model_ar.forecast(horizon=1)[0]
                    y_pred_ma = self.model_ma.forecast(horizon=1)[0]
                    y_pred_arma = self.model_arma.forecast(horizon=1)[0]

                    mae_score_lr.update(y_true=y, y_pred=np.exp(y_pred_lr))
                    mae_score_rf.update(y_true=y, y_pred=np.exp(y_pred_rf))
                    mae_score_ar.update(y_true=y, y_pred=np.exp(y_pred_ar))
                    mae_score_ma.update(y_true=y, y_pred=np.exp(y_pred_ma))
                    mae_score_arma.update(y_true=y, y_pred=np.exp(y_pred_arma))

                    print('----------------------------------------------------------------------------------------------')
                    print("At " + str(date_time_stamp) +  "  : " )
                    print( "                                        Linear Regresion     |    Random Forest     |    Moving Average     |    Auto regressive     |    ARMA ")
                    print("***** The real price: " + str(round(y,3)) +  " The predicted price: " + str(round(np.exp(y_pred_lr),3))  + "     |    " + str(round(np.exp(y_pred_rf),3))  +  "    |    " + str(round(np.exp(y_pred_ma),3)) +"    |    " + str(round(np.exp(y_pred_ar),3)) + "    |    " +  str(round(np.exp(y_pred_arma),3))  )
                    print('----------------------------------------------------------------------------------------------')
                    for key,value in self.models_dict.items():
                        value_list = self.scoresmae[key].copy()
                        value_list.append(eval('mae_score_' + value).get())
                        self.scoresmae.update({key:value_list})

                    for key,value in self.models_dict.items():
                        value_list = self.close_price_pred[key].copy()
                        value_list.append(np.exp(eval('y_pred_' + value)))
                        self.close_price_pred.update({key:value_list})



                    axs[0][0].plot(timestamps[2:],self.close_price_stream[2:], color='green', label='Stream data', linewidth=4)
                    axs[0][0].plot(timestamps[2:],self.close_price_pred[self.models_list[0]][2:], color='magenta', label=self.models_list[0])
                    axs[0][1].plot(timestamps[2:], self.close_price_pred[self.models_list[1]][2:], color='red',label=self.models_list[1])
                    axs[0][1].plot(timestamps[2:], self.close_price_stream[2:], color='green', label='Stream data', linewidth=4)
                    axs[0][2].plot(timestamps[2:], self.close_price_pred[self.models_list[2]][2:], color='blue',label=self.models_list[2])
                    axs[0][2].plot(timestamps[2:], self.close_price_stream[2:], color='green', label='Stream data', linewidth=4)
                    axs[0][3].plot(timestamps[2:], self.close_price_pred[self.models_list[3]][2:], color='orange',label=self.models_list[3])
                    axs[0][3].plot(timestamps[2:], self.close_price_stream[2:], color='green', label='Stream data',linewidth=4)
                    axs[0][4].plot(timestamps[2:], self.close_price_pred[self.models_list[4]][2:], color='black', label=self.models_list[4])
                    axs[0][4].plot(timestamps[2:], self.close_price_stream[2:], color='green', label='Stream data', linewidth=4)

                    axs[1][0].plot(timestamps[2:],self.scoresmae[self.models_list[0]], color='magenta', label=self.models_list[0])
                    axs[1][1].plot(timestamps[2:], self.scoresmae[self.models_list[1]], color='red',label=self.models_list[1])
                    axs[1][2].plot(timestamps[2:], self.scoresmae[self.models_list[2]], color='blue', label=self.models_list[2])
                    axs[1][3].plot(timestamps[2:], self.scoresmae[self.models_list[3]], color='orange', label=self.models_list[3])
                    axs[1][4].plot(timestamps[2:], self.scoresmae[self.models_list[4]], color='black', label=self.models_list[4])

                    axs[0][0].tick_params(axis='x', labelrotation=45)
                    axs[0][1].tick_params(axis='x', labelrotation=45)
                    axs[0][2].tick_params(axis='x', labelrotation=45)
                    axs[0][3].tick_params(axis='x', labelrotation=45)
                    axs[0][4].tick_params(axis='x', labelrotation=45)
                    axs[1][0].tick_params(axis='x', labelrotation=45)
                    axs[1][1].tick_params(axis='x', labelrotation=45)
                    axs[1][2].tick_params(axis='x', labelrotation=45)
                    axs[1][3].tick_params(axis='x', labelrotation=45)
                    axs[1][4].tick_params(axis='x', labelrotation=45)

                    if not is_plotted:
                        axs[0][0].legend()
                        axs[0][1].legend()
                        axs[0][2].legend()
                        axs[0][3].legend()
                        axs[0][4].legend()
                        axs[1][0].legend()
                        axs[1][1].legend()
                        axs[1][2].legend()
                        axs[1][3].legend()
                        axs[1][4].legend()
                        axs[0][0].set_ylabel('Predictions')
                        axs[0][1].set_ylabel('Predictions')
                        axs[0][2].set_ylabel('Predictions')
                        axs[0][3].set_ylabel('Predictions')
                        axs[0][4].set_ylabel('Predictions')
                        axs[1][0].set_ylabel('MAE score')
                        axs[1][1].set_ylabel('MAE score')
                        axs[1][2].set_ylabel('MAE score')
                        axs[1][3].set_ylabel('MAE score')
                        axs[1][4].set_ylabel('MAE score')

                        is_plotted = True


                    self.model_lr.learn_one(x, np.log(y))
                    self.model_rf.learn_one(x, np.log(y))
                    self.model_ma.learn_one(np.log(y))
                    self.model_ar.learn_one(np.log(y))
                    self.model_arma.learn_one(np.log(y))
                    plt.pause(1)

                i += 1





#trade_online = TrainModelOnline(company_name = 'BNP.PA', topic_name = 'BNP.PA_stocks' )
#trade_online.train_model_online()

