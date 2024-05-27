import csv
import yfinance as yf
from finta import TA
import pandas as pd
from pymongo import MongoClient

connect = 'mongodb://host.docker.internal:27017/'
# connect = 'mongodb://localhost:27017/'

def load_csv_to_mongodb(csv_file, database_name, collection_name):
    client = MongoClient(connect)
    db = client[database_name]
    collection = db[collection_name]

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            collection.insert_one(row)
    print(f"Data imported from {csv_file} to {collection_name} collection in {database_name} database.")


def load_bitcoin_mongodb(database_name, collection_name):
    client = MongoClient(connect)
    db = client[database_name]
    collection = db[collection_name]

    # Download data
    stock = 'BTC-USD'
    start = '2014-09-17'
    end = '2024-02-13'

    df = yf.download(stock, start, end)
    df.index = df.index.date
    df.index = df.index.map(str)
    df.fillna(0, inplace=True)
    df['RSI'] = TA.RSI(df, 12)
    df['SMA'] = TA.SMA(df)
    df['OBV'] = TA.OBV(df)
    df = df.fillna(0)

    # Convert DataFrame to dictionary format
    df_dict = df.reset_index().to_dict("records")

    # Insert data into MongoDB
    collection.delete_many({})  # Clear existing data
    collection.insert_many(df_dict)  # Insert new data

    print("lstm data successfully saved to MongoDB")

def load_lstm_mongodb(database_name, collection_name):
    file_path = 'spambase.data'
    columns = [
        "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our", "word_freq_over",
        "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail", "word_freq_receive",
        "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
        "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit", "word_freq_your",
        "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl", "word_freq_george",
        "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857", "word_freq_data",
        "word_freq_415", "word_freq_85", "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm",
        "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
        "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
        "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average",
        "capital_run_length_longest", "capital_run_length_total", "label"
    ]
    data = pd.read_csv(file_path, header=None, names=columns)

    documents = data.to_dict(orient='records')

    client = MongoClient(connect)
    db = client[database_name]
    collection = db[collection_name]

    collection.insert_many(documents)
    print("rnn data successfully saved to MongoDB")

load_csv_to_mongodb('dummy_data.csv', 'blog', 'gradient')
load_bitcoin_mongodb('blog', 'btc_usd')
load_lstm_mongodb('blog', 'rnn')