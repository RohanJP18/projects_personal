import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
msoft = pd.read_csv("Microsoft Stock Data.csv")
msoft['Date'] = pd.to_datetime(msoft['Date'])
msoft.set_index('Date', inplace=True)

# Use only 'Close' price for training
data = msoft[['Close']].values  

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Convert time series to supervised learning format
def create_sequences(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])  # Sequence
        Y.append(data[i+seq_length])    # Target (next time step)
    return np.array(X), np.array(Y)

seq_length = 60  # 60 days history for prediction
X, Y = create_sequences(data_scaled, seq_length)

# Split data into training and testing sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)  # Predict next stock price
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, Y_train, epochs=20, batch_size=16, validation_data=(X_test, Y_test), verbose=1)

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Inverse transform predictions
train_preds = scaler.inverse_transform(train_preds)
test_preds = scaler.inverse_transform(test_preds)

# Inverse transform actual prices
Y_train_actual = scaler.inverse_transform(Y_train)
Y_test_actual = scaler.inverse_transform(Y_test)

# Create plot traces for visualization
trace_train_actual = go.Scatter(
    x=msoft.index[seq_length:split+seq_length], 
    y=Y_train_actual.flatten(), 
    mode='markers', 
    name='Train Actual'
)
trace_train_pred = go.Scatter(
    x=msoft.index[seq_length:split+seq_length], 
    y=train_preds.flatten(), 
    mode='lines', 
    name='Train Predicted'
)

trace_test_actual = go.Scatter(
    x=msoft.index[split+seq_length:], 
    y=Y_test_actual.flatten(), 
    mode='markers', 
    name='Test Actual'
)
trace_test_pred = go.Scatter(
    x=msoft.index[split+seq_length:], 
    y=test_preds.flatten(), 
    mode='lines', 
    name='Test Predicted'
)

# Define layout
layout = go.Layout(
    title="Stock Price Prediction with LSTM",
    xaxis=dict(title='Date'),
    yaxis=dict(title='Stock Price')
)

# Create figure and show plot
plot_lstm = go.Figure(data=[trace_train_actual, trace_train_pred, trace_test_actual, trace_test_pred], layout=layout)
plot_lstm.show()
