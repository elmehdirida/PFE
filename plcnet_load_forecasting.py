# Correcting the code for the PLCNet model to ensure it's working with the provided dataset

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, concatenate, Dropout, Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error ,r2_score

# Load and preprocess data
def load_data(filename):
    data = pd.read_csv(filename, sep=';')
    data['time'] = pd.to_datetime(data['time'], format='%m/%d/%y %I:%M %p')
    data.set_index('time', inplace=True)
    data['load'] = pd.to_numeric(data['load'], errors='coerce')
    data = data.dropna()

    X = []
    y = []
    for i in range(len(data) - 144):
        x = data['load'].iloc[i:i+144].values.reshape(144, 1)
        y_val = data['load'].iloc[i+144]
        X.append(x)
        y.append(y_val)

    X = np.array(X)
    y = np.array(y)

    return X, y

# PLCNet model
def create_plcnet_model(input_shape):
    # CNN Path
    cnn_input = Input(shape=input_shape)
    cnn = Conv1D(64, 2, activation='relu')(cnn_input)
    cnn = MaxPooling1D()(cnn)
    cnn = Conv1D(32, 2, activation='relu')(cnn)
    cnn_flattened = Flatten()(cnn)

    # LSTM Path
    lstm_input = Input(shape=input_shape)
    lstm = LSTM(48, activation='relu')(lstm_input)

    # Merge and Fully Connected Layers
    merged = concatenate([lstm, cnn_flattened])
    merged_reshaped = Reshape((1, -1))(merged) # Reshape for LSTM compatibility
    merged_lstm = LSTM(300, activation='relu')(merged_reshaped)
    merged_lstm = Dropout(0.3)(merged_lstm)
    dense = Dense(units=128, activation='sigmoid')(merged_lstm)
    output = Dense(units=1, activation='sigmoid')(dense)

    model = tf.keras.models.Model(inputs=[cnn_input, lstm_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Load and preprocess the data
X, y = load_data('hourly_load_data.csv')

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Create the PLCNet model
model = create_plcnet_model(X_scaled.shape[1:])

# Train the model
model.fit([X_scaled, X_scaled], y_scaled, epochs=10, verbose=1)

# Make predictions
predictions_scaled = model.predict([X_scaled, X_scaled])

# Inverse transform the predictions to original scale
predictions_original_scale = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

# Calculate and print MSE
mse = mean_squared_error(y, predictions_original_scale)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y, predictions_original_scale)
print(f'R-squared value: {r2}')

prediction_percentage = 100 * (1 - mse / np.var(y))
print(f'Prediction Percentage: {prediction_percentage:.2f}%')