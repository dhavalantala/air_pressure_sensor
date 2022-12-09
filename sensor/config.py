import pymongo
import os, sys
from dataclasses import dataclass

@dataclass
class EnvironmentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL")


env_var = EnvironmentVariable()

TARGET_COLUMN_MAPPING = {
    "pos":1,
    "neg":0
}

mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
TARGET_COLUMN = "class"