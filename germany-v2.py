import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, concatenate, Dropout, Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Load and preprocess data 
def load_data(filename, input_days, output_days):
    data = pd.read_csv(filename, sep=',')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data['Consumption'] = pd.to_numeric(data['Consumption'], errors='coerce')
    data = data.dropna()

    # Filter by year for training and testing
    train_data = data[(data['Date'].dt.year >= 2012) & (data['Date'].dt.year <= 2015)]
    test_data = data[(data['Date'].dt.year >= 2016) & (data['Date'].dt.year <= 2017)]

    X_train, y_train = create_xy(train_data, input_days, output_days)
    X_test, y_test = create_xy(test_data, input_days, output_days)

    return X_train, y_train, X_test, y_test

# Helper function
def create_xy(data, input_days, output_days):
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
    dense1 = Dense(units=128, activation='sigmoid')(merged_lstm) # Sigmoid activation function
    dense2 = Dense(units=64, activation='sigmoid')(dense1) # Sigmoid activation function
    output = Dense(units=output_units, activation='sigmoid')(dense2) # Sigmoid activation function

    model = tf.keras.models.Model(inputs=[cnn_input, lstm_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Load and preprocess the data
X_train, y_train, X_test, y_test = load_data('opsd_germany_daily.csv', input_days=7, output_days=1)

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
y_test_scaled = scaler_y.transform(y_test)

# Create the PLCNet model
model = create_plcnet_model(input_shape=(7, 1), output_units=1)

# Train the model
model.fit([X_train_scaled, X_train_scaled], y_train_scaled, epochs=300)

# Make predictions
predictions_scaled = model.predict([X_test_scaled, X_test_scaled])

# Inverse transform the predictions to original scale
predictions_original_scale = scaler_y.inverse_transform(predictions_scaled)

# Calculate and print metrics
mse = mean_squared_error(y_test, predictions_original_scale)
mape = mean_absolute_percentage_error(y_test, predictions_original_scale)
r2 = r2_score(y_test, predictions_original_scale)
prediction_percentage = 100 * (1 - mse / np.var(y_test))

print(f'MAPE (Mean Absolute Percentage Error): {mape * 100:.2f}%')
print(f'R-squared value: {r2}')
print(f'Prediction Percentage: {prediction_percentage:.2f}%')
