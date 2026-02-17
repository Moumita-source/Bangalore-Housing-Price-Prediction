import sys
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import FeatureEngineerArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from src.entity.estimator import MyModel
from src.utils.main_utils import save_object
from src.constants import SCHEMA_FILE_PATH, TARGET_COLUMN

class ModelTrainer:
    def __init__(self, feature_engineering_artifact: FeatureEngineerArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param feature_engineering_artifact: Output reference of feature engineering artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.feature_engineering_artifact = feature_engineering_artifact
        self.model_trainer_config = model_trainer_config
        self._schema_config =read_yaml_file(file_path=SCHEMA_FILE_PATH)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)    
        
    def get_preprocessor_object(self) -> ColumnTransformer:
        """
        Returns the preprocessor object
        """    
        preprocessor = ColumnTransformer(
                transformers= [
                    ('numerical', StandardScaler(), self._schema_config["processed_numerical_columns"]),
                    ('categorical', OneHotEncoder(handle_unknown='ignore', drop='first'), self._schema_config["processed_categorical_columns"])
                ]
            ) 
        
        return preprocessor

    def get_model_object_and_report(self, train: pd.DataFrame, test: pd.DataFrame, preprocessor: ColumnTransformer) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training RandomForestRegressor with specified parameters")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train.drop(columns = [TARGET_COLUMN]), train[TARGET_COLUMN], test.drop(columns = [TARGET_COLUMN]), test[TARGET_COLUMN]
            logging.info("train-test split done.")

            # Create preprocessor
            
            
            # Create random forest regressor model
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(
                    random_state = self.model_trainer_config._random_state,
                    max_depth = self.model_trainer_config._max_depth,
                    max_features = self.model_trainer_config._max_features,
                    min_samples_leaf = self.model_trainer_config._min_samples_leaf,
                    min_samples_split = self.model_trainer_config._min_samples_split,
                    n_estimators = self.model_trainer_config._n_estimators,
                    n_jobs = -1
                ))
            ])
            
            # Fit the model
            logging.info("Model training going on....")
            model.fit(X= x_train, y= y_train)
            logging.info("Model training done....")
            
            # Predictions and evaluation metrics
            y_pred = model.predict(x_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2_score_value = r2_score(y_test, y_pred)
            
            # Creating metric artifact
            metric_artifact = RegressionMetricArtifact(rmse= rmse, r2_score= r2_score_value)
            return model, metric_artifact
            
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            featured_train_df = self.read_data(file_path= self.feature_engineering_artifact.featured_train_file_path)
            featured_test_df = self.read_data(file_path= self.feature_engineering_artifact.featured_test_file_path)
            logging.info("train-test data loaded")
            
            # Load preprocessing object
            preprocessor = self.get_preprocessor_object()
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=featured_train_df, test=featured_test_df, preprocessor = preprocessor)
            logging.info("Model object and artifact loaded.")
            
            # Check if models r2_score meets the expected threshold
            if metric_artifact.r2_score < ModelTrainerConfig.expected_r2_score:
                logging.info("No model found with r2 score above the base score")
                raise Exception("No model found with r2 score above the base score")
            
            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object= preprocessor, trained_model_object= trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")
            
            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path= self.model_trainer_config.trained_model_file_path,
                metric_artifact= metric_artifact
            )
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise MyException(e, sys) from e