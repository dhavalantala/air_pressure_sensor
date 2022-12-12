import sys, os
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.predictor import ModelResolver
from sensor.utils import load_object
import pandas as pd
import numpy as np
from datetime import datetime

PREDICTION_DIR = "prediction"
PREDICTION_FILE_NAME = f"{datetime.now().strftime('_%m_%d_%Y__%H_%M_%S')}.csv"

def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info("Reading the file: {input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na":np.NAN}, inplace=True)
        
        # Validation

        logging.info(f"Loading tranformer to transform datset")
        transformer = load_object(file_path= model_resolver.get_latest_transformer_path())
        
        input_feature_names = list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])
        
        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)

        logging.info(f"Target encoder to convert predicted column to categorical column")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())
        cat_prediction = target_encoder.inverse_transform(prediction)

        df["prediction"] = prediction
        df["cat_prediction"] = cat_prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".csv", PREDICTION_FILE_NAME)
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)
        df.to_csv(prediction_file_path, index=False, header=True)
        return prediction_file_path
    except Exception as e:
        raise SensorException(e, sys)