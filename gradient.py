import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from pymongo import MongoClient
import json
connect = 'mongodb://host.docker.internal:27017/'
# connect = 'mongodb://localhost:27017/'

def grad_random(database_name, collection_name):
    lgb_model = lgb.Booster(model_file='lgb_model.txt')
    scaler = joblib.load('scaler_lgb.save')

    client = MongoClient(connect)
    db = client[database_name]
    collection = db[collection_name]

    data = list(collection.find())
    data_dict = [json.loads(json.dumps(item, default=str)) for item in data]
    data_df = pd.DataFrame(data_dict)
    data_df['time_spent'] = data_df['time_spent'].astype(int)
    random_index = np.random.randint(0, len(data_df))

    data_new = data_df[['age', 'gender', 'time_spent', 'platform']]
    pd.set_option('display.max_columns', 50)

    data_processed = pd.get_dummies(data_new, columns=['gender', 'platform'])

    data_processed_new = data_processed[
        ['age', 'gender_female', 'time_spent', 'gender_male', 'gender_non-binary', 'platform_Facebook',
         'platform_Instagram', 'platform_YouTube']]

    random_data_processed = data_processed_new.iloc[random_index]
    X_random = random_data_processed.drop('time_spent').to_frame().T
    X_random_scaled = scaler.transform(X_random)

    # Прогнозирование на случайном примере с помощью загруженной модели
    y_random_pred = lgb_model.predict(X_random_scaled, verbose=0)
    return random_data_processed.to_dict(), y_random_pred[0]


def study_grad(database_name, collection_name):
    client = MongoClient(connect)
    db = client[database_name]
    collection = db[collection_name]
    data = list(collection.find())
    data_dict = [json.loads(json.dumps(item, default=str)) for item in data]
    data_df = pd.DataFrame(data_dict)

    data_new = data_df[['age', 'gender', 'time_spent', 'platform']]
    data_new['age'] = data_new['age'].astype(int)
    data_new['time_spent'] = data_new['time_spent'].astype(int)
    pd.set_option('display.max_columns', 50)

    data_processed = pd.get_dummies(data_new, columns=['gender', 'platform'])

    data_processed_new = data_processed[
        ['age', 'gender_female', 'time_spent', 'gender_male', 'gender_non-binary', 'platform_Facebook',
         'platform_Instagram', 'platform_YouTube']]

    # Разделение данных на предикторы (X) и целевую переменную (y)
    X = data_processed_new.drop('time_spent', axis=1)
    y = data_new['time_spent']

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Создание объекта Dataset для LightGBM
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

    # Параметры модели
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    # Обучение модели
    num_round = 100
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=True)]
    lgb_model = lgb.train(params, train_data, num_boost_round=num_round, valid_sets=[test_data], callbacks=callbacks)

    # Прогнозирование на тестовом наборе
    y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    def calculate_mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_mask = y_true != 0
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        return mape

    mape_score = calculate_mape(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    lgb_model.save_model('lgb_model.txt')
    joblib.dump(scaler, 'scaler_lgb.save')

    return f"MAPE: {mape_score} Mean Squared Error: {mse}"

