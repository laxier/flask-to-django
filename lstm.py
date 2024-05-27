from pymongo import MongoClient
import joblib
import os
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from tensorflow.keras.metrics import MeanAbsoluteError
connect = 'mongodb://host.docker.internal:27017/'
# connect = 'mongodb://localhost:27017/'

def study_lstm(epos, database_name, collection_name):
    # Connect to MongoDB
    client = MongoClient(connect)
    db = client[database_name]
    collection = db[collection_name]

    # Retrieve data from MongoDB
    data = list(collection.find())

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.set_index('index', inplace=True)
    df.index = pd.to_datetime(df.index)

    # Fill NaN values with 0 (although data should already be clean)
    df = df.select_dtypes(include=np.number)
    df.fillna(0, inplace=True)

    # Ensure the dataframe is sorted by index (date)
    df.sort_index(inplace=True)

    # Prepare the data for LSTM model
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # Prepare test data
    test_data = df_scaled['Close'][int(0.8 * len(df)) - 100:].values
    scaler_test = MinMaxScaler()
    scaled_test_data = scaler_test.fit_transform(test_data.reshape(-1, 1))

    x_test, y_test = [], []

    for i in range(100, len(test_data)):
        x_test.append(scaled_test_data[i - 100:i, 0])
        y_test.append(scaled_test_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Prepare training data
    training_data = df_scaled['Close'][:int(0.8 * len(df))].values
    scaled_training_data = scaler.fit_transform(training_data.reshape(-1, 1))

    x_train, y_train = [], []

    for i in range(100, len(training_data)):
        x_train.append(scaled_training_data[i - 100:i, 0])
        y_train.append(scaled_training_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build and compile the LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=60, return_sequences=True),
        Dropout(0.3),
        LSTM(units=80, return_sequences=True),
        Dropout(0.4),
        LSTM(units=120, return_sequences=False),
        Dropout(0.5),
        Dense(units=1, activation='linear')
    ])

    # Обучение модели
    # model.compile(optimizer='adam', loss='mean_squared_error')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError()])

    # Определение функции обратного вызова для отправки прогресса обучения через сокеты
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    # Обучение модели с использованием функции обратного вызова SocketCallback
    history = model.fit(x_train, y_train, epochs=epos, batch_size=64, verbose=0, validation_data=(x_test, y_test),
                        callbacks=[early_stopping])
    y_predict = model.predict(x_test, verbose=0)
    model.save("model_linear0.keras")
    # close_prices = df[['Close']].values
    # scaler = MinMaxScaler(feature_range=(0,1))
    # scaler.fit(close_prices)  # close_prices - это ваш массив данных для обучения scaler
    # joblib.dump(scaler, 'scaler_fit.save')
    # joblib.dump(scaler, 'btc_price_scaler.pkl')

    # #для R^2
    r2 = r2_score(y_test, y_predict)
    loss, mse = model.evaluate(x_test, y_test, verbose=0)
    # print(f'Test MSE: {mse}')
    # print(f'R²: {r2}')
    return f'Test MSE: {mse} R^2: {r2}'


def predict_day_price(date, loaded_model, df, scaler, sequence_length=100):
    # Преобразование введенной даты в формат datetime
    date = pd.to_datetime(date)

    # Находим индекс даты в DataFrame

    # Выборка последовательности цен закрытия за последние sequence_length дней перед указанной датой
    start_index = df.index.get_loc(date) - sequence_length
    end_index = df.index.get_loc(date)
    last_sequence = df['Close'][start_index:end_index].values

    # Нормализация последовательности
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))

    # Изменение формы последовательности для предсказания
    last_sequence_scaled = last_sequence_scaled.reshape((1, sequence_length, 1))

    # Предсказание цены на следующий день
    predicted_price_scaled = loaded_model.predict(last_sequence_scaled, verbose=0)

    # Обратное масштабирование предсказанной цены
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    return predicted_price[0][0]


def get_real_pericted(start_date, end_date, db_name, collection_name):
    client = MongoClient(connect)
    db = client[db_name]
    collection = db[collection_name]

    current_path = os.getcwd()
    file_path = os.path.join(current_path, 'model_linear0.keras')
    loaded_model = load_model(file_path)
    file_path = os.path.join(current_path, 'btc_price_scaler.pkl')
    scaler = joblib.load(file_path)

    df = pd.DataFrame(list(collection.find()))
    df['Date'] = pd.to_datetime(df['index'])
    df.drop(columns=['_id', 'index'], inplace=True)
    df.set_index('Date', inplace=True)

    predicted_prices = []
    real_prices = []

    df_time = df.loc[start_date:end_date]

    for date in df_time.index:
        predicted_price = predict_day_price(date, loaded_model, df, scaler)
        predicted_prices.append(predicted_price)
        real_price = df.loc[df.index == date]['Close'].values[0]
        real_prices.append(real_price)

    df_results = pd.DataFrame({'Date': df_time.index, 'Real Price': real_prices, 'Predicted Price': predicted_prices})
    return df_results