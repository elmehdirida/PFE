import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dropout, LSTM, Dense, Conv1D, MaxPooling1D, concatenate, Flatten, Reshape
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filename):
    data = pd.read_csv(filename)
    data = data.drop_duplicates()
    data['Sum'] = data['Sum'].astype('float32')

    X = []
    y = []
    for i in range(len(data) - 24):
        x = data['Sum'].iloc[i:i+24].values.reshape(24, 1)
        y_val = data['Sum'].iloc[i+24]
        X.append(x)
        y.append(y_val)

    X = np.array(X)
    y = np.array(y)

    return X, y

def plcnet(X, y):
    input_layer = Input(shape=(X.shape[1], X.shape[2]))

    lstm_branch = LSTM(48, activation='relu', return_sequences=True)(input_layer)
    lstm_branch = LSTM(32, activation='relu')(lstm_branch)

    cnn_branch = Conv1D(64, 2, activation='relu')(input_layer)
    cnn_branch = MaxPooling1D()(cnn_branch)
    cnn_branch = Conv1D(32, 2, activation='relu')(cnn_branch)
    cnn_branch = Flatten()(cnn_branch)

    merged = concatenate([lstm_branch, cnn_branch])
    merged = Reshape((1, -1))(merged)  # Reshape to match the ndim=3 expected by LSTM
    merged = LSTM(300, activation='relu', return_sequences=True)(merged)
    merged = Dropout(0.3)(merged)
    merged = LSTM(200, activation='relu')(merged)

    dense1 = Dense(128, activation='relu')(merged)
    dense2 = Dense(64, activation='relu')(dense1)

    output_layer = Dense(1, activation='relu')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer='adam')

    model.fit(X, y, epochs=150)

    return model

if __name__ == '__main__':
    X, y = load_data('power-consumption.csv')  # Replace with your actual file path

    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    model = plcnet(X_scaled, y_scaled)

    # Make predictions
    predictions_scaled = model.predict(X_scaled)

    # Inverse transform the predictions to original scale
    predictions_original_scale = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Additional evaluation metrics
    mse = mean_squared_error(y, predictions_original_scale)
    print(f'Mean Squared Error: {mse}')

    r2 = r2_score(y, predictions_original_scale)
    print(f'R-squared value: {r2}')

    prediction_percentage = 100 * (1 - mse / np.var(y))
    print(f'Prediction Percentage: {prediction_percentage:.2f}%')
