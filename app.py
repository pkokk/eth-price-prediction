from binance.client import Client
from datetime import datetime,timedelta,date
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import  matplotlib.pyplot as plt
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

#Load .env file with parameters
load_dotenv(find_dotenv())
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")

def download_dataset(years,api_key=API_KEY,api_secret=API_SECRET):
    #create client for binance api
    client = Client(api_key, api_secret)

    #create start date for our api call 
    years_back = years
    date_start = 1000* int((datetime.now() - timedelta(days=years_back*365)).timestamp())

    #return candles of ETH prices aggregated per day. 
    bars = client.get_historical_klines('ETHUSDT', '1d', date_start)
    #we only keep close price and date and convert date from milliseconds epoch to date 
    for line in bars:
        del line[5:]
        del line[1:4]
        line[0]=datetime.fromtimestamp(line[0]/1000.0).date()


    df = pd.DataFrame(bars, columns=['date','price'])
    df.set_index('date', inplace=True)
    return df

def fit_arima(train,order):
    model = ARIMA((train),order = order)
    model = model.fit()
    return model

df = download_dataset(3)
days_pred = 5
df[['price']] = df[['price']].applymap(lambda x: 0.0 if pd.isnull(x) else float(x))
index_future_dates = [(datetime.today() + timedelta(days=x)).date() for x in range(days_pred)]
df = df.reindex(df.index.union(index_future_dates))


st.title("ARIMA close price prediction ETH USD")    

st.sidebar.write("## Choose ARIMA parameters")
p = st.sidebar.slider("Choose parameter p ", 0 , 15)
d = st.sidebar.slider("Choose parameter d ", 0 , 15)
q = st.sidebar.slider("Choose parameter q ", 0 , 15)

model = fit_arima(df['price'],order = (p,d,q))


pred= model.predict(start = 0, end=len(df)+days_pred).rename('predictinos')
df['pred'] = pred

st.line_chart(df[-50:])

mapelist = ((df.iloc[:-days_pred].price-df.iloc[:-days_pred].pred).abs()
                 .div(df.iloc[:-days_pred].price)
                 .cumsum()
                 /np.arange(1,len(df)-days_pred+1)
             )

mape = round(np.array(mapelist).mean(),4)

st.write("**Mean Absolute Percentage Error : {}**".format(str(mape)))

