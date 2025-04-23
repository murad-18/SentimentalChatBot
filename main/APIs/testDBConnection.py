# test_mongo_connection.py

import os
from pymongo import MongoClient, errors

def test_connection(uri: str):
    try:
        # Try to connect with a short timeout
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Will throw if it can’t connect
        info = client.server_info()
        print("Connected to MongoDB Atlas!")
        print("   Server version:", info.get("version"))
        print("   Databases available:", client.list_database_names())
    except errors.ServerSelectionTimeoutError as e:
        print("Failed to connect to MongoDB Atlas:")
        print("   ", e)
    finally:
        client.close()

if __name__ == "__main__":
    # Either set MONGO_URI in your env, or paste it here:
    # for passwords with special characters use: 
    # from urllib.parse import quote_plus # using this library we can encode our password and use it in url
    # raw_pwd = "your_raw_password"
    # encoded_pwd = quote_plus(raw_pwd)
    uri = "mongodb+srv://username:password@cluster0.twzfxsg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    test_connection(uri)
