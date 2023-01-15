from data_streaming import stock_stream
from model_train_online import TrainModelOnline
import ray
ray.init()

company_name   = input('Please enter the company name: ')

@ray.remote
def launch_stock_stream(company_name):
    stock_stream(company_name=company_name)

@ray.remote
def launch_online_learning(company_name):
    trade_online = TrainModelOnline(company_name=company_name, topic_name=company_name+'_stocks')
    trade_online.train_model_online()

ray.get([launch_stock_stream.remote(company_name), launch_online_learning.remote(company_name)])
