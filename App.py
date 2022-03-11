import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import numpy as np 
import pandas as pd

### Data Collection
start="2020-01-01"
today=date.today().strftime("%Y-%m-%d")
st.title('WELCOME! BMS Institute of Technology & Management : STOCK PRICE PREDICTION')

stocks=('ADANIPOWER.NS','AAPL','GOOG','MSFT','GME')

selected_stock=st.selectbox('select the company stocks',stocks)

n_years=st.slider('number of days prediction:',1,100)
period=n_years*1

@st.cache
def load_data(ticker):
    data=yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


df1=data.reset_index()['Close']

### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 
from sklearn.preprocessing import MinMaxScaler
 
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


#f=st.slider('No. of days in past', 1, 30)
f=18

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = f
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)



# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout
from tensorflow.keras.layers import Dense
import math
from sklearn.metrics import mean_squared_error

### Create the Stacked LSTM model

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(f,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')




model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=25,batch_size=5,verbose=1)



### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)



### Calculate RMSE performance metrics

math.sqrt(mean_squared_error(y_train,train_predict))


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


import matplotlib.pyplot as plt 

### Plotting 
# shift train predictions for plotting
look_back=f
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


x_input=test_data[(len(test_data)-f):].reshape(1,-1)
x_input.shape


temp_input=list(x_input)
temp_input=temp_input[0].tolist()



# demonstrate prediction for next 10 days


from numpy import array

Z = 1
lst_output=[]
n_steps=f
i=0
while(i<int(Z)):
    
    if(len(temp_input)>f):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

lst_output

#plot prediction price
def plot_pred_data():
     gif=go.Figure()
     gif.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="actual value"))
     gif.add_trace(go.Scatter(x=data['Date'], y=trainPredictPlot , name="predicted price"))
     gif.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
     st.plotly_chart(gif)
     
plot_pred_data()

day_new=np.arange(1,f+1)
day_pred=np.arange((f+1),(f+1)+int(Z))


plt.plot(day_new,scaler.inverse_transform(df1[(len(df1)-f):]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


day_pred=np.arange(1,int(Z)+1)
plt.plot(day_pred,scaler.inverse_transform(lst_output))



#print(scaler.inverse_transform(lst_output))
st.write(scaler.inverse_transform(lst_output))
