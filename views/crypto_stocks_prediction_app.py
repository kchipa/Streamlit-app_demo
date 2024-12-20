import yfinance as yf
import pandas as pd
import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model



st.title("Crypto/Stocks prediction")


stock = st.text_input("Enter the Stock ID", "META")

from datetime import datetime
end= datetime.now()
start = datetime(end.year-20, end.month, end.day)

meta_data = yf.download(stock, start, end)

# Flatten the MultiIndex columns to keep only the price type
meta_data.columns = meta_data.columns.get_level_values(0)

model = "https://github.com/kchipa/Streamlit-app_demo/blob/main/views/Latest_stock_price_model.keras"
#model = load_model('views\Latest_stock_price_model.keras') 
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

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])
    
x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 },
    index = meta_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data.head())

st.subheader('Original Close Price and Predictions for 250 days')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([meta_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(['Data- not used', 'Original Test Data', 'Predicted Test data'])
st.pyplot(fig)

st.subheader("Future Close Price value")

last_100 = meta_data[['Close']].tail(100)
last_100 = scaler.fit_transform(last_100['Close'].values.reshape(-1,1)).reshape(1,-1,1)
last_100 = last_100.tolist()

def predict_future(no_of_days, prev_100, scaler):
    future_prediction = []

    # Ensure prev_100 is a NumPy array and has the correct shape
    prev_100 = np.array(prev_100).reshape(1, 100, -1)  # Adjust the last dimension as needed

    for i in range(no_of_days):
        next_day = model.predict(prev_100)  # Predict the next day
        next_day_inverse = scaler.inverse_transform(next_day)  # Inverse transform the prediction
        future_prediction.append(next_day_inverse[0][0])  # Append the predicted value directly

        # Update prev_100 for the next prediction
        # Create a new array with the predicted value appended
        new_data = np.array(next_day).reshape(1, 1, -1)  # Reshape to (1, 1, features)
        prev_100 = np.concatenate((prev_100[:, 1:, :], new_data), axis=1)  # Keep the last 99 and add the new prediction

    return future_prediction

no_od_days = int(st.text_input("How many?: ", "10"))
future_results = predict_future(no_od_days, last_100, scaler)

future_results = np.array(future_results).reshape(-1,1)
fig = plt.figure()
plt.plot(pd.DataFrame(future_results), marker='o')
for i in range(len(future_results)):
    plt.text(i,future_results[i], int(future_results[i][0]))
plt.xlabel('Future Days')
plt.ylabel('Close Price')
plt.title('Future Close price of BTC')
st.pyplot(fig)
