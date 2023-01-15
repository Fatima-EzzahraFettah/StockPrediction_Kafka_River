import yfinance as yf
from datetime import date
import json
import time
from kafka_producer import producer



def stock_stream(company_name: str):
    topic_name = company_name + '_stocks'
    currentdate = date.today()
    while True:
        try:
            data = yf.download(company_name,start=currentdate,interval='1m')
            data = data.reset_index(drop=False)
            data['Datetime'] = data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

            my_dict1 = data.iloc[-1].to_dict()
            my_dict2 = data.iloc[-2].to_dict()
            print('Waiting for the next stock price (60 seconds)')
            time.sleep(60)
            if my_dict2['Datetime'] != my_dict1['Datetime']:
                print(my_dict1)
                msg = json.dumps(my_dict1)
                producer.send(topic_name, key= b'Stock Update ', value=msg.encode())
                print(f"Producing to {topic_name}")
                producer.flush()
        except:
            pass
