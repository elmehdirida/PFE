# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:30:13 2023

@author: HP
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

class PLCNet(tf.keras.Model):
    def __init__(self, lstm_units, cnn_filters, kernel_size, output_units):
        super().__init__()

        self.lstm_branch = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=True),
            tf.keras.layers.LSTM(lstm_units)
        ])

        self.cnn_branch = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu'),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=lstm_units)
        ])

        self.attention = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(units=output_units)

    def call(self, inputs):
        lstm_output = self.lstm_branch(inputs)
        cnn_output = self.cnn_branch(inputs)

        combined_output = self.attention([lstm_output, cnn_output])
        output = self.dense(combined_output)

        return output

# Load the dataset (replace 'opsd_germany_daily.csv' with the actual file path)
#data = pd.read_csv('opsd_germany_daily.csv')
data = pd.read_csv('power-consumption.csv')
# Extract target variable
target = data['Sum']

# Normalize the target variable
scaler_target = StandardScaler()
target_normalized = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Create sequences for LSTM and CNN branches
seq_length = 24  # Number of past time steps to consider
sequences = []
for i in range(len(target_normalized) - seq_length):
    seq = target_normalized[i:i + seq_length]
    sequences.append(seq)

# Convert sequences to tensors
sequences = np.array(sequences)
sequences = np.squeeze(sequences, axis=-1)
sequences = tf.convert_to_tensor(sequences, dtype=tf.float32)

# Reshape sequences to include the third dimension
sequences = tf.reshape(sequences, (sequences.shape[0], sequences.shape[1], 1))

# Split data into training, validation, and testing sets
train_size = int(0.7 * len(sequences))
val_size = int(0.2 * len(sequences))

train_data = sequences[:train_size]
val_data = sequences[train_size:train_size + val_size]
test_data = sequences[train_size + val_size:]

# Split target into corresponding sets
train_target = target.values[seq_length:train_size + seq_length]  # Adjust for the sequence length
val_target = target.values[train_size + seq_length:train_size + val_size + seq_length]
test_target = target.values[train_size + val_size + seq_length:]

# Normalize the target variables
train_target_normalized = scaler_target.fit_transform(train_target.reshape(-1, 1))
val_target_normalized = scaler_target.transform(val_target.reshape(-1, 1))
test_target_normalized = scaler_target.transform(test_target.reshape(-1, 1))


# Define the model
model = PLCNet(lstm_units=64, cnn_filters=32, kernel_size=3, output_units=1)
# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])

# Train the model
model.fit(train_data, train_target_normalized, epochs=24, validation_data=(val_data, val_target_normalized))

# Evaluate the model
eval_result = model.evaluate(test_data, test_target_normalized)
print("Evaluation Result (MSE, MAPE):", eval_result)

# Make predictions
predictions = model.predict(test_data)

# Inverse transform the predictions to original scale
predictions_original_scale = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()


r2 = r2_score(test_target, predictions_original_scale)
print(f'R-squared value: {r2}')

