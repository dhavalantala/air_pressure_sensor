from sensor.entity import artifact_entity,config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from typing import Optional
from sklearn.pipeline import Pipeline
import pandas as pd
from sensor import utils
import numpy as np
from dataclasses import dataclass
from sensor.config import TARGET_COLUMN
import os,sys 
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder


class DataTransformation:

    def __init__(self, 
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                    data_transformation_config:config_entity.DataTransformationConfig):

        try:
            logging.info(f"{'>>'*20} Data Tansformation {'<<'*20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_transformation_object(cls):
        try:
            simple_imputer = SimpleImputer(strategy = "constant", fill_value=0)
            robust_scaler = RobustScaler()
            
            constant_pipeline = Pipeline(steps=[
                ('Imputer', simple_imputer),
                ('RobustScaler', robust_scaler)
            ])
            return constant_pipeline
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            logging.info(f"reading training and testing file")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            logging.info(f"Selecting input feature for train and test dataframe")
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            logging.info(f"Selecting target feature for train and test dataframe")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info(f"Transformation on target columns")
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            
            logging.info(f"Transformation on input feature columns")
            transformation_pipeline = DataTransformation.get_transformation_object()
            transformation_pipeline.fit(input_feature_train_df)
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target: {target_feature_train_arr}")
            smt = SMOTETomek(sampling_strategy="minority")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target: {target_feature_train_arr}")


            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target: {target_feature_test_arr}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target: {target_feature_test_arr}")
            
            logging.info(f"Concinate the feature and Target encoder")
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info(f"Save numpy array data")
            utils.save_numpy_array_data(file_path = self.data_transformation_config.transformed_train_path, array = train_arr)
            utils.save_numpy_array_data(file_path = self.data_transformation_config.transformed_test_path, array = test_arr)

            logging.info(f"save the transformer model and label encoder model")
            utils.save_object(file_path = self.data_transformation_config.transform_object_path, obj = transformation_pipeline)
            utils.save_object(file_path = self.data_transformation_config.target_encoder_path, obj = label_encoder)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                                        transform_object_path = self.data_transformation_config.transform_object_path,
                                        transformed_train_path = self.data_transformation_config.transformed_train_path,
                                        transformed_test_path = self.data_transformation_config.transformed_test_path,
                                        target_encoder_path = self.data_transformation_config.target_encoder_path
            )

            logging.info(f"Data transformation object {data_transformation_artifact}")

            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)
