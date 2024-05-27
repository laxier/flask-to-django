import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from keras.models import load_model

connect = 'mongodb://host.docker.internal:27017/'
# connect = 'mongodb://localhost:27017/'

def study_rnn(epos, database_name, collection_name):
    # Connect to MongoDB
    client = MongoClient(connect)
    db = client[database_name]
    collection = db[collection_name]

    # Retrieve data from MongoDB
    cursor = collection.find()
    data = pd.DataFrame(list(cursor))
    data = data.drop('_id', axis=1)

    # Split data into features and labels
    features = data.drop('label', axis=1)
    labels = data['label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Define RNN model
    model = Sequential([
        Reshape((1, -1), input_shape=(57,)),
        LSTM(64, activation='tanh', return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Define custom callback to emit progress updates

    # Train the model
    history = model.fit(X_train, y_train, epochs=epos, validation_data=(X_test, y_test), verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Save the trained model
    model.save("model_rnn1.keras")

    return f'Accuracy on test set: {accuracy:.4f}'


def predict_random_spam(data, model):
    random_sample = data.sample(n=1)
    features = random_sample.drop('label', axis=1)
    label = random_sample['label'].iloc[0]
    prediction = model.predict(features, verbose=0)
    predicted_label = "Спам" if prediction[0][0] > 0.5 else "Не спам"
    actual_label = "Спам" if label == 1 else "Не спам"
    return f'Реальная метка: {actual_label}, Предсказанная метка: {predicted_label}'


def predict_spam(num_samples, database_name, collection_name):
    client = MongoClient(connect)
    db = client[database_name]
    collection = db[collection_name]

    # Retrieve data from MongoDB
    cursor = collection.find()
    data = pd.DataFrame(list(cursor))
    data = data.drop('_id', axis=1)
    ans = []

    model = load_model('model_linear0.keras')
    for i in range(num_samples):
        ans.append(predict_random_spam(data, model))
    return ans


# study_rnn(10, 'blog', 'rnn')
# print(predict_spam(10, 'blog', 'rnn'))
