import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

stock_tweets = '~/stoch_tweets.csv'
data_sentiment = '~/data_sentiment.csv'
stock_yfinance_data = '~/stock_yfinance_data.csv'
#Getting Historical Stock Data
#fetch historical stock data using the `yfinance` library.
stock_tweets_data = pd.read_csv('stock_tweets.csv')
stock_yfinance_data = pd.read_csv('stock_yfinance_data.csv')
sentiment_data = pd.read_csv('data_sentiment.csv')
sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])
sentiment_data['Date'] = sentiment_data['Date'].dt.strftime('%d/%m/%Y')
#Merging on 'date' to align the sentiment with the corresponding
merged_data = pd.merge(stock_yfinance_data, sentiment_data, on=['Stock Name'])
dataset = pd.read_csv('stock_tweets.csv')
#we need to pick two coulmns ( date, sentiment result) for training_dataset
training_set = merged_data[['positive', 'negative', 'Close']].values
print(training_set)
#normalization
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
#Incorporating Timesteps Into Data
#We should input our data in the form of a 3D array to the LSTM model.
# Define X_train and y_train as lists
X_train = []
y_train = []

# Iterate over the data
for i in range(60, 64479):
    # Append the sequence of 60 values to X_train
    X_train.append(training_set_scaled[i-60:i, 0])
    # Append the next value to y_train
    y_train.append(training_set_scaled[i, 0])

# Convert lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape X_train to have 3 dimensions
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#Creating the LSTM Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
# Define the LSTM Model architecture
model = Sequential()
# Adding the first LSTM layer and some Dropout regularization
# Input shape should match the shape of our training dataset
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

#second layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Adding a Dense layer to output 1 value (the predicted stock price)
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training/fiting the model with the training set
model.fit(X_train, y_train, epochs=54, batch_size=32)

# Predicting the Stock Prices using Test Data
# Split the data into training and testing sets
training_size = int(len(training_set_scaled) * 0.8)
test_data = training_set_scaled[training_size - 60:, :]  # Including last 60 days of training data for test set

X_test = []
y_test = training_set_scaled[training_size:, 2]  # Actual stock prices for the test set

# Creating test data with timesteps
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Making Predictions
predicted_stock_price = model.predict(X_test)

# Inverse Scaling for Prediction
predicted_stock_price = sc.inverse_transform(np.concatenate((test_data[60:, :2], predicted_stock_price), axis=1))[:, 2]

# Evaluate the Model
rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))

print(f'Root Mean Squared Error: {rmse}')

# Visualize the Results
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.plot(merged_data['Date'][training_size:], y_test, color = 'red', label = 'Real Stock Price')
plt.plot(merged_data['Date'][training_size:], predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.xticks(np.arange(0, len(merged_data['Date'][training_size:]), step=int(len(merged_data['Date'][training_size:])/10)))
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
