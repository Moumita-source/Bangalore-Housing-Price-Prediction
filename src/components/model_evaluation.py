import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.entity.s3_estimator import HousingEstimator
from dataclasses import dataclass

from src.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.exception import MyException
from src.logger import logging
from src.constants import TARGET_COLUMN, LOCATION_THRESHOLD, CONVERSION_FACTORS
from src.utils.main_utils import load_json_object, load_object
from sklearn.metrics import r2_score

@dataclass
class EvalauteModelResponse:
    trained_model_r2_score: float
    best_model_r2_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:
    
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,data_transformation_artifact: DataTransformationArtifact, model_trainer_artifact: ModelTrainerArtifact,
                 model_evaluation_config: ModelEvaluationConfig):
        
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
    
    def extract_bhk(self, x):
        """
        Extract numeric from the string
        """
        try:
            x = str(x).lower()
            if 'rk' in x:
                return 1
            return int(x.split()[0])
        except:
            return None
        
    def convert_total_sqft(self, x):
        """
        Convert total_sqft string to float in square feet.
        Handles ranges and unit conversions.
        """  
        if isinstance(x, float):
            return x
        
        x = str(x).strip()
        # Case 1: range like "1133 - 1384"
        if "-" in x:
            tokens = x.split("-")
            if len(tokens) == 2:
                try:
                    return (float(tokens[0]) + float(tokens[1])) / 2
                except:
                    None
                    
        # Case 2: single number with unit 
        x = x.replace(",", "") # remove commas 
        for unit, factor in CONVERSION_FACTORS.items():
            if unit in x.lower():
                num_part = x.lower().replace(unit, "").strip() 
                try: 
                    return float(num_part) * factor 
                except:
                    return None 
                
        try:
            return float(x)
        except:
            return None     
    
    def process_location_feature(self, df):
        """
        Transformations on location feature
        1. Fill the missing values with Other
        2. Format the location values
        3. Extract feature location_grouped where we take the features with high frequency location and keep remaining as other
        """
        df["location"] = df["location"].fillna("other")
        df["location"] = df["location"].str.strip().str.lower()
        location_counts = df["location"].value_counts()
        top_locations = location_counts[location_counts >= int(LOCATION_THRESHOLD)].index.tolist()
        df["location_grouped"] = df["location"].apply(lambda x: x if x in top_locations else 'other') 
        return df    
    
    def process_size_feature(self, df):
        """
        Transformations on size feature
        1. Extract the numeric value from the size feature
        2. Fill the missing values in extracted feature with its median
        3. Drop the original size feature
        """
        df["bhk"] = df["size"].apply(self.extract_bhk)
        median_bhk = df["bhk"].median()
        df["bhk"] = df["bhk"].fillna(median_bhk).astype(int)
        df = df.drop(["size"], axis = 1)
        return df
    
    def process_society_feature(self, df):
        """
        Transformation on society feature
        1. Extract a binary feature has_society from society feature
        """
        df["has_society"] = df["society"].notnull().astype(int)
        df = df.drop(["society"], axis = 1)
        return df
    
    def process_bath_feature(self, df):
        """
        Transformation on bath feature
        1. Replace all the missing values of bath feature with that of its corresponding bhk feature
        """
        df.loc[df["bath"].isnull(), "bath"] = df.loc[df["bath"].isnull(), "bhk"]
        return df  
    
    def process_balcony_feature(self, df):
        """
        Transformation on balcony feature
        1. Fill the missing values in balcony with its median
        """
        df["balcony"] = df["balcony"].fillna(df["balcony"].median())
        return df
    
    def process_availability_feature(self, df):
        """
        Transformation on availability feature
        1. Drop the availability feature as it didnt serve as important feature in modelling later
        """
        df = df.drop("availability", axis = 1)
        return df
    
    def process_total_sqft_feature(self, df):
        """
        Transformation on total_sqft feature
        1. Extract the numeric values from the total_sqrt with various case conversion into its sqft unit
        """
        df["total_sqft_num"] = df["total_sqft"].apply(self.convert_total_sqft)
        median_sqft = df["total_sqft_num"].median() 
        df["total_sqft_num"] = df["total_sqft_num"].fillna(median_sqft)
        df = df.drop(columns=["total_sqft"])
        df = df.rename(columns={"total_sqft_num": "total_sqft"})
        return df
    
    def apply_target_encoding(self, x: pd.DataFrame, mapping_file_path: str, col: str) -> pd.DataFrame:
        """
        Apply target encoding to a given column using a saved mapping + global mean.
    
        Parameters:
        - x: DataFrame to transform
        - mapping_file_path: path to JSON file containing {"mapping": ..., "global_mean": ...}
        - col: column name to encode
    
        Returns:
        - DataFrame with new encoded column added
        """
        loaded_info = load_json_object(mapping_file_path)
        mapping = loaded_info["mapping"]
        global_mean = loaded_info["global_mean"]
        
        x[col + "_target_enc"] = x[col].map(mapping).fillna(global_mean)
        return x
    
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
    
    def get_best_model(self) -> Optional[HousingEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_evaluation_config.bucket_name
            model_path = self.model_evaluation_config.s3_model_key_path
            house_estimator = HousingEstimator(bucket_name= bucket_name,
                                               model_path= model_path)
            
            if house_estimator.is_model_present(model_path= model_path):
                return house_estimator
            return None
        except Exception as e:
            raise MyException(e, sys) from e
     
    def evaluate_model(self) -> EvalauteModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis = 1), test_df[TARGET_COLUMN]
            
            logging.info("Test data loaded and now transforming it for prediction ...")
            
            x = self.process_location_feature(x)
            x = self.process_size_feature(x)
            x = self.process_society_feature(x)
            x = self.process_bath_feature(x)
            x = self.process_balcony_feature(x)
            x = self.process_availability_feature(x)
            x = self.process_total_sqft_feature(x)
            x = self.apply_target_encoding(x, self.data_transformation_artifact.target_encoded_mapping_file_path,"location")
            x = self.construct_number_of_bathrooms_per_bhk(x)
            x = self.construct_total_sqrt_per_bhk(x)
            x = self.construct_extra_bathrooms(x)
            x = self.construct_total_sqft_log(x)
            
            logging.info("Reached here")
            
            trained_model_r2_score = self.model_trainer_artifact.metric_artifact.r2_score
            logging.info(f"R2 score for this model : {trained_model_r2_score}")
            
            best_model_r2_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing r2 score for production model...")
                y_hat_best_model = best_model.predict(x)
                logging.info(f"Successfully predicted here")
                best_model_r2_score = r2_score(y, y_hat_best_model)
                logging.info(f"R2 score Production model: {best_model_r2_score}, R2 score new trained model : {trained_model_r2_score}")
                
            tmp_best_model_score = 0 if best_model_r2_score is None else best_model_r2_score
            result = EvalauteModelResponse(
                trained_model_r2_score= trained_model_r2_score,
                best_model_r2_score= best_model_r2_score,
                is_model_accepted= trained_model_r2_score > tmp_best_model_score,
                difference= trained_model_r2_score - tmp_best_model_score
            )    
            
            logging.info(f"Result: {result}")
            return result
                
        except Exception as e:
            raise MyException(e, sys) from e     
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_evaluation_config.s3_model_key_path
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted= evaluate_model_response.is_model_accepted,
                s3_model_path= s3_model_path,
                trained_model_path= self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy= evaluate_model_response.difference
            )
            
            logging.info(f"Model evaluation artifact : {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise MyException(e, sys) from e             
