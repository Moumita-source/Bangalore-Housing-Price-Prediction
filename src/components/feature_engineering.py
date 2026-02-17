import os
import sys
import pandas as pd
import numpy as np

from src.entity.config_entity import FeatureEngineerConfig
from src.entity.artifact_entity import FeatureEngineerArtifact, DataTransformationArtifact
from src.exception import MyException
from src.logger import logging



class FeatureEngineering:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 feature_engineer_config: FeatureEngineerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.feature_engineer_config = feature_engineer_config
        except Exception as e:
            raise MyException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        
    
    def construct_number_of_bathrooms_per_bhk(self, df) -> pd.DataFrame:
        """
        Create new feature number of bathrooms per bhk
        """
        df["bath_per_bhk"] = df["bath"] / df["bhk"]
        return df
    
    def construct_total_sqrt_per_bhk(self, df) -> pd.DataFrame:
        """
        Create number of square feet per bhk
        """
        df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"]
        return df
    
    def construct_extra_bathrooms(self, df) -> pd.DataFrame:
        """
        Create a flag if number of bathrooms exceeded the number of rooms
        """
        df["extra_bath"] = (df["bath"] > df["bhk"] + 1).astype(int)
        return df
    
    def construct_total_sqft_log(self, df) -> pd.DataFrame:
        """
        Capture non-linear of total-sqft
        """
        df["total_sqft_log"] = np.log1p(df["total_sqft"])
        return df
    
    def initiate_feature_engineering(self) -> FeatureEngineerArtifact:
        """
        Initiates the feature engineering component for the pipeline.
        """    
        try:
            logging.info("Feature Engineering Started !!!")
            
            # Load transformed train and test data
            transformed_train_df = self.read_data(file_path= self.data_transformation_artifact.transformed_train_file_path)
            transformed_test_df = self.read_data(file_path= self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Transformed Train-Test data loaded")
            
            logging.info("Constructing features on train data")
            transformed_train_df = self.construct_number_of_bathrooms_per_bhk(transformed_train_df)
            transformed_train_df = self.construct_total_sqrt_per_bhk(transformed_train_df)
            transformed_train_df = self.construct_extra_bathrooms(transformed_train_df)
            transformed_train_df = self.construct_total_sqft_log(transformed_train_df)
            
            logging.info("Constructing features on test data")
            transformed_test_df = self.construct_number_of_bathrooms_per_bhk(transformed_test_df)
            transformed_test_df = self.construct_total_sqrt_per_bhk(transformed_test_df)
            transformed_test_df = self.construct_extra_bathrooms(transformed_test_df)
            transformed_test_df = self.construct_total_sqft_log(transformed_test_df)
            
            logging.info("Feature engineering completed")
            os.makedirs(self.feature_engineer_config.feature_engineer_dir,exist_ok=True)
            
            logging.info(f"Exporting feature engineered train and test file path.")
            transformed_train_df.to_csv(self.feature_engineer_config.featured_train_file_path,index=False,header=True)
            transformed_test_df.to_csv(self.feature_engineer_config.featured_test_file_path,index=False,header=True)
            
            return FeatureEngineerArtifact(
                featured_train_file_path= self.feature_engineer_config.featured_train_file_path,
                featured_test_file_path= self.feature_engineer_config.featured_test_file_path
            )
              
        except Exception as e:
            raise MyException(e, sys) from e
            
            
        
        
        
      