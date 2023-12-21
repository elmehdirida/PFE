# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:25:06 2023

@author: HP
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

class PLCNet(tf.keras.Model):
    def __init__(self, cnn_filters1, cnn_filters2, lstm_units, lstm_dense_units, output_units):
        super(PLCNet, self).__init__()
        
        self.lstm_branch = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, activation='relu')
        ])

        self.cnn_branch = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=lstm_units)
        ])


        # Merged Layer
        self.merge_layer = tf.keras.layers.Concatenate()

        # LSTM-Dense Path
        self.lstm_dense_path = tf.keras.Sequential([
            tf.keras.layers.LSTM(300, activation='relu', return_sequences=True),
            tf.keras.layers.LSTM(300, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(lstm_dense_units, activation='relu'),
        ])

        # Final Dense Layers
        self.dense1 = tf.keras.layers.Dense(128, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(output_units, activation='sigmoid')

    def call(self, inputs):
        cnn_output = self.cnn_branch(inputs)
        lstm_output = self.lstm_branch(inputs)

        merged_data = self.merge_layer([cnn_output, lstm_output])
        lstm_dense_output = self.lstm_dense_path(tf.reshape(merged_data, (-1, 96, 1)))
        final_output1 = self.dense1(lstm_dense_output)
        final_output2 = self.dense2(final_output1)

        return final_output2

# Load the dataset (replace 'hourly_load_data.csv' with the actual file path)
data = pd.read_csv('hourly_load_data.csv', sep=';', parse_dates={'datetime': ['time']}, index_col='datetime')

# Extract target variable
target = data['load']

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
model = PLCNet(cnn_filters1=64, cnn_filters2=32, lstm_units=48, lstm_dense_units=300, output_units=1)
# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])

# Train the model
model.fit(train_data, train_target_normalized, epochs=24, validation_data=(val_data, val_target_normalized))

# Evaluate the model
eval_result = model.evaluate(test_data, test_target_normalized)
print("Evaluation Result (MSE, mape):", eval_result)

# Make predictions
predictions = model.predict(test_data)

# Inverse transform the predictions to original scale
predictions_original_scale = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Additional evaluation metrics
mse = mean_squared_error(test_target, predictions_original_scale)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(test_target, predictions_original_scale)
print(f'R-squared value: {r2}')

prediction_percentage = 100 * (1 - mse / np.var(test_target))
print(f'Prediction Percentage: {prediction_percentage:.2f}%')
