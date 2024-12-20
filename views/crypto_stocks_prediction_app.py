import yfinance as yf
import pandas as pd
import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import requests
import os
import tensorflow as tf


st.title("Crypto/Stocks prediction")

stock = st.text_input("Enter the Stock ID", "META")

from datetime import datetime
end= datetime.now()
start = datetime(end.year-20, end.month, end.day)

meta_data = yf.download(stock, start, end)

# Flatten the MultiIndex columns to keep only the price type
meta_data.columns = meta_data.columns.get_level_values(0)


#model = load_model('Steamlit-app_demo/views/Latest_stock_price_model.keras') 
st.subheader("Stock Data")
st.write(meta_data)
st.write(meta_data.head())
splitting_len = int(len(meta_data)*0.7)
x_test = pd.DataFrame(meta_data['Close'][splitting_len:])
st.write(meta_data.columns)

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig 
    
st.subheader('Original Close Price and MA for 250 days')
meta_data['MA_for_250_days'] = meta_data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15, 6), meta_data['MA_for_250_days'], meta_data,0))

st.subheader('Original Close Price and MA for 200 days')
meta_data['MA_for_200_days'] = meta_data['Close'].rolling(200).mean()
st.pyplot(plot_graph((15, 6), meta_data['MA_for_200_days'], meta_data,0))

st.subheader('Original Close Price and MA for 100 days')
meta_data['MA_for_100_days'] = meta_data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15, 6), meta_data['MA_for_100_days'], meta_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), meta_data['MA_for_100_days'], meta_data,1,meta_data['MA_for_250_days']))


