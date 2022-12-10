import os, sys
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity import config_entity, artifact_entity
from sensor import utils
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

class ModelTrainer:

    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Train {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def train_model(self, x, y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x, y)
            return xgb_clf 
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading Training and Testing Array")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(f"Train the model")
            model = self.train_model(x=x_train, y=y_train)

            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_train, yhat_train)

            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_test, yhat_test)         

            logging.info(f"Train Score: {f1_train_score}, Test Score: {f1_test_score}")

            # Check for overfitting or underfitting oe expected score
            logging.info(f"Checking if our model is underfitted or not")
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give expected accuracy: {self.model_trainer_config.expected_score}, Model Actual Score: {f1_test_score}")
            
            diff = abs(f1_train_score - f1_test_score)

            logging.info(f"Checking if our model is overfitted or not")
            if diff > self.model_trainer_config.overfitting_thres:
                raise Exception(f"Train and test Score diff: {diff} is more than the Overfitting threshold {self.model_trainer_config.overfitting_thres}")

            # Save the trained model
            logging.info(f"Save the Train model with f1 score: {f1_test_score}")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj = model)

            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path = self.model_trainer_config.model_path, 
                                                                        f1_train_score = f1_train_score, 
                                                                        f1_test_score = f1_test_score)

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)
