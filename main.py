from sensor.logger import logging
from sensor.exception import SensorException
import sys, os
from sensor.pipeline.training_pipeline import start_training_pipeline
from sensor.pipeline.batch_prediction import start_batch_prediction

file_path = "/Users/dhavalantala/Desktop/air_pressure_sensor/artifact/12_12_2022__16_09_20/data_ingestion/12_12_2022__16_09_20/dataset/test.csv"

if __name__ == "__main__":
    try:
        # start_training_pipeline()
        output_file  = start_batch_prediction(input_file_path= file_path)
        print(output_file)
    except Exception as e:
        print(e)