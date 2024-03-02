import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, concatenate, Dropout, Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_percentage_error

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

# Create the PLCNet model
model = create_plcnet_model(input_shape=(7, 1), output_units=1)

# Train the model
model.fit([X_scaled, X_scaled], y_scaled, epochs=70)

# Make predictions
predictions_scaled = model.predict([X_scaled, X_scaled])

# Inverse transform the predictions to original scale
predictions_original_scale = scaler_y.inverse_transform(predictions_scaled)


# Calculate rmse
mse = mean_squared_error(y, predictions_original_scale)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.2f}')

# Calculate mape
mape = mean_absolute_percentage_error(y, predictions_original_scale)

print(f'MAPE (Mean Absolute Percentage Error): {mape * 100:.2f}%')

r2 = r2_score(y, predictions_original_scale)
print(f'R-squared value: {r2}')


prediction_percentage = 100 * (1 - mse / np.var(y))
print(f'Prediction Percentage: {prediction_percentage:.2f}%')