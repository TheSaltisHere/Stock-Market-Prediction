# libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# get data from yahoo of any company
data_frame = web.DataReader(
    'AMZN', data_source='yahoo', start='2010-01-01', end='2022-01-31')
# see the data
data_frame
#rows and columns
data_frame.shape
# see how the data looks like graphically
data_frame['Close'].plot(figsize=(16, 8))
# get the close price of the stock data in a numpy array
data_close = data_frame.filter(['Close'])
data_close_set = data_close.values
# training set 80% of the data
training_data_len = math.ceil(len(data_close)*0.8)
training_data_len

# scaling  the data
scale = MinMaxScaler(feature_range=(0, 1))
scaled_data = scale.fit_transform(data_close)
scaled_data
training_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()
# convert training datasets to numpy and reshaping the data
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape
# lstm model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,
          input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
# running the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)
# creating test-data set
test_data = scaled_data[training_data_len-60:, :]
x_test = []
y_test = data_close_set[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
# make data into numpy
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# Get the predicted price of the stock
predictions = model.predict(x_test)
predictions = scale.inverse_transform(predictions)
# get root mean squared error
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
rmse

# plot data
train = data_close[:training_data_len]
valid = data_close[training_data_len:]
valid['Predictions'] = predictions

# visualize the data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price USD ($)', fontsize=18)
plt.plot(train['Close'], linewidth=3)
plt.plot(valid['Close'], color='Green', linewidth=3, label='Closed Prices')
plt.plot(valid['Predictions'], color='red', linewidth=3, label='Predictions')
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

valid
# Analysis on last 60 days to predict the price of the stock
stock_data = web.DataReader(
    'AMZN', data_source='yahoo', start='2021-01-01', end='2022-01-31')
stock_data_close = stock_data.filter(['Close'])
stock_data_close_set = stock_data_close[-60:].values
days_scaled = scale.transform(stock_data_close_set)
X_test_sample = []
X_test_sample.append(days_scaled)
X_test_sample = np.array(X_test_sample)
X_test_sample = np.reshape(
    X_test_sample, (X_test_sample.shape[0], X_test_sample.shape[1], 1))
predicted_price = model.predict(X_test_sample)
predicted_price = scale.inverse_transform(predicted_price)
print(predicted_price)

stock_data_actual = web.DataReader(
    'AMZN', data_source='yahoo', start='2022-01-31', end='2022-01-31')
print(stock_data_actual['Close'])


# accuracy
print("Accuracy of the model is:")
error_percent = abs(
    (predicted_price[0][0]-stock_data_actual['Close'][0])/stock_data_actual['Close'][0])*100
print(100-error_percent)
