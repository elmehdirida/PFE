# Complete code with validation split and early stopping for the PLCNet model

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, concatenate, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Load and preprocess data
def load_data(filename, input_days, output_days):
    data = pd.read_csv(filename, sep=',')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data.set_index('Date', inplace=True)
    data['Consumption'] = pd.to_numeric(data['Consumption'], errors='coerce')
    data = data.dropna()

    X = []
    y = []
    for i in range(len(data) - input_days - output_days + 1):
        x = data['Consumption'].iloc[i:i+input_days].values.reshape(input_days, 1)
        y_val = data['Consumption'].iloc[i+input_days:i+input_days+output_days]
        X.append(x)
        y.append(y_val)

    X = np.array(X)
    y = np.array(y)

    return X, y

# PLCNet model
def create_plcnet_model(input_shape, output_units):
    cnn_input = Input(shape=input_shape)
    cnn = Conv1D(64, 2, activation='relu')(cnn_input)
    cnn = MaxPooling1D()(cnn)
    cnn = Conv1D(32, 2, activation='relu')(cnn)
    cnn_flattened = Flatten()(cnn)

    lstm_input = Input(shape=input_shape)
    lstm = LSTM(48, activation='relu')(lstm_input)

    merged = concatenate([lstm, cnn_flattened])
    merged_reshaped = Reshape((1, -1))(merged)
    merged_lstm = LSTM(300, activation='relu')(merged_reshaped)
    merged_lstm = Dropout(0.3)(merged_lstm)
    dense = Dense(units=256, activation='relu')(merged_lstm)
    output = Dense(units=output_units, activation='relu')(dense)

    model = tf.keras.models.Model(inputs=[cnn_input, lstm_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Load and preprocess the data
X, y = load_data('opsd_germany_daily.csv', input_days=7, output_days=1)

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y)

# Split the data into training and validation sets
split_index = int(len(X_scaled) * 0.8)  # 80% for training, 20% for validation
X_train, X_val = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_val = y_scaled[:split_index], y_scaled[split_index:]

# Create the PLCNet model
model = create_plcnet_model(input_shape=(7, 1), output_units=1)

# Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Train the model with validation data and early stopping
model.fit([X_train, X_train], y_train, epochs=200, validation_data=([X_val, X_val], y_val), callbacks=[early_stopping], verbose=1)

# Evaluate the model on the validation set
val_predictions_scaled = model.predict([X_val, X_val])
val_predictions_original_scale = scaler_y.inverse_transform(val_predictions_scaled)

val_mse = mean_squared_error(y_val, val_predictions_original_scale)
val_rmse = np.sqrt(val_mse)
val_mape = mean_absolute_percentage_error(y_val, val_predictions_original_scale)
val_r2 = r2_score(y_val, val_predictions_original_scale)

# Print the validation metrics
print(f'Validation Root Mean Squared Error: {val_rmse}')
print(f'Validation MAPE (Mean Absolute Percentage Error): {val_mape * 100:.2f}%')
print(f'Validation R-squared value: {val_r2}')
