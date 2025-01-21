import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID","GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-15,end.month,end.day)

data = yf.download(stock,start,end)

model = load_model("latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(data)

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None) :
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'orange')
    plt.plot(full_data.Close, 'b')
    if extra_data :
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 Days')
data['MA_250'] = data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15,6),data['MA_250'],data,0))

st.subheader('Original Close Price and MA for 200 Days')
data['MA_200'] = data['Close'].rolling(200).mean()
st.pyplot(plot_graph((15,6),data['MA_200'],data,0))

st.subheader('Original Close Price and MA for 100 Days')
data['MA_100'] = data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15,6),data['MA_100'],data,0))


st.subheader('Original Close Price and MA for 250 Days and MA for 100 Days')

st.pyplot(plot_graph((15,6),data['MA_250'],data,1,data['MA_100']))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'])

x_data=[]
y_data=[]

for i in range(100,len(scaled_data)):
  x_data.append(scaled_data[i-100:i])
  y_data.append(scaled_data[i])


x_data , y_data  = np.array(x_data) , np.array(y_data)

splitting_len= int(len(x_data)*0.7)
x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]
x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]


predictions = model.predict(x_test)

inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test)

ploting_data = pd.DataFrame(
    {
      'Actual':inv_y_test.reshape(-1),
      'Predicted':inv_predictions.reshape(-1)
    } , 
    index = data.index[splitting_len+100:]
)

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader("Original Close Price vs Predicted Close Price")
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([data.Close[:splitting_len+100],ploting_data],axis=0))
plt.legend( ["Data not used", "Original Test data", "Predicted Test data"] )
st.pyplot(fig)
