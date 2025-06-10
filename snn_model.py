import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import snntorch as snn
from torch.utils.data import DataLoader, TensorDataset

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

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

# Create DataLoader
train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# Define SNN model
class SNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape input to (batch_size * sequence_length, input_dim)
        batch_size, seq_length, _ = x.shape
        x = x.view(-1, seq_length)  # Reshape to (batch_size * sequence_length, input_dim)
        
        # Initialize membrane potential
        mem1 = self.lif1.init_leaky()
        
        # Pass through the first layer
        spk1, mem1 = self.lif1(self.fc1(x), mem1)
        
        # Pass through the second layer (non-spiking)
        output = self.fc2(spk1)
        
        # Reshape output to (batch_size, output_dim)
        output = output.view(batch_size, -1)
        
        return output

input_dim = seq_length
hidden_dim = 50
output_dim = 1

model = SNNModel(input_dim, hidden_dim, output_dim)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predictions
model.eval()
with torch.no_grad():
    train_preds = model(X_train).numpy()
    test_preds = model(X_test).numpy()

# Inverse transform predictions
train_preds = scaler.inverse_transform(train_preds)
test_preds = scaler.inverse_transform(test_preds)

# Inverse transform actual prices
Y_train_actual = scaler.inverse_transform(Y_train.numpy())
Y_test_actual = scaler.inverse_transform(Y_test.numpy())

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
    title="Stock Price Prediction with SNN",
    xaxis=dict(title='Date'),
    yaxis=dict(title='Stock Price')
)

# Create figure and show plot
plot_snn = go.Figure(data=[trace_train_actual, trace_train_pred, trace_test_actual, trace_test_pred], layout=layout)
plot_snn.show()
