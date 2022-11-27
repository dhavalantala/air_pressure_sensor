import pymongo
import pandas as pd
import json

# provide the mongoDb localhost url to connect python to mongoDB
client = pymongo.MongoClient("mongodb+srv://mydatabase:RBLF6oCgx2tQ3veF@cluster0.cqvrdck.mongodb.net/?retryWrites=true&w=majority")


# data file path
DATA_FILE_PATH = "/Users/dhavalantala/Desktop/air_pressure_sensor/aps_failure_training_set1.csv"
DATABASE_NAME = "aps"
COLLECTION_NAME = "sensor"

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    # print(f"Rows and columns: {df.shape}")

    # convert DataFrame io json format so that we can ump this record in MongoDB
    df.reset_index(drop=True, inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    # print(json_record[0])

    # insert converted json record to MongoDB 
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)