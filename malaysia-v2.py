# Description: PLCNet model for load forecasting
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, concatenate, Dropout, Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_percentage_error

# Load and preprocess data 
def load_data(filename, input_hours, output_hours):
    data = pd.read_csv(filename, sep=';')
    data['time'] = pd.to_datetime(data['time'], format='%m/%d/%y %I:%M %p')
    data['load'] = pd.to_numeric(data['load'], errors='coerce')
    data = data.dropna()
    # Filter by year for training and testing
    train_data = data[data['time'].dt.year == 2009]  # Training set: 2009
    test_data = data[data['time'].dt.year == 2010]  # Testing set: 2010

    # Create X and y for training 
    X_train, y_train = create_xy(train_data, input_hours, output_hours)

    # Create X and y for testing
    X_test, y_test = create_xy(test_data, input_hours, output_hours)

    return X_train, y_train, X_test, y_test

# Helper function (you likely already have this)
def create_xy(data, input_hours, output_hours):
    X = []
    y = []
    for i in range(len(data) - input_hours - output_hours + 1):
        x = data['load'].iloc[i:i+input_hours].values.reshape(input_hours, 1)
        y_val = data['load'].iloc[i+input_hours:i+input_hours+output_hours]
        X.append(x)
        y.append(y_val)
    return np.array(X), np.array(y)


def create_plcnet_model(input_shape, output_units):
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
    merged_lstm = Dropout(0.3)(merged_lstm) # 30% dropout to avoid overfitting
    dense1 = Dense(units=128, activation='sigmoid')(merged_lstm) # Sigmoid activation function
    dense2 = Dense(units=64, activation='sigmoid')(dense1) # Sigmoid activation function
    output = Dense(units=output_units, activation='sigmoid')(dense2) # Sigmoid activation function

    model = tf.keras.models.Model(inputs=[cnn_input, lstm_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

X_train, y_train, X_test, y_test = load_data('hourly_load_data.csv', input_hours=24, output_hours=1)

# Normalize the data (apply normalization separately to training and testing)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
y_train_scaled = scaler_y.fit_transform(y_train)  
X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)  # Use fitted scaler_X
y_test_scaled = scaler_y.transform(y_test)

# Create the PLCNet model
model = create_plcnet_model(input_shape=(24, 1), output_units=1)

# Train the model (use the training data)
model.fit([X_train_scaled, X_train_scaled], y_train_scaled, epochs=20)

predictions_scaled = model.predict([X_test_scaled, X_test_scaled])

# Inverse transform the predictions and the real y test
predictions_original_scale = scaler_y.inverse_transform(predictions_scaled)
y_test_original_scale = scaler_y.inverse_transform(y_test_scaled) 

# Calculate metrics
mse = mean_squared_error(y_test_original_scale, predictions_original_scale) # Use y_test_original_scale
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')
mape = mean_absolute_percentage_error(y_test_original_scale, predictions_original_scale) 
print(f'MAPE (Mean Absolute Percentage Error): {mape * 100:.2f}%')

r2 = r2_score(y_test_original_scale, predictions_original_scale)
print(f'R-squared value: {r2}')

prediction_percentage = 100 * (1 - mse / np.var(y_test_original_scale)) # Use y_test_original_scale
print(f'Prediction Percentage: {prediction_percentage:.2f}%')