import os
import sys
import pandas as pd

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, LOCATION_THRESHOLD, CONVERSION_FACTORS, TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, save_json_object, save_target_encoded_json_object


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
      
      
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
        
    def remove_bhk_outliers(self, df):
        exclude_indices = []
    
        for location, loc_df in df.groupby('location'):
            # Compute stats for each BHK in this location
            bhk_stats = loc_df.groupby('bhk')['price'].agg(['mean', 'std', 'count'])
        
            # Iterate over BHK groups
            for bhk, sub_df in loc_df.groupby('bhk'):
                if (bhk - 1) in bhk_stats.index:
                    stats = bhk_stats.loc[bhk - 1]
                    if stats['count'] > 5:
                        exclude = sub_df[sub_df['price'] < (stats['mean'] - 0.5 * stats['std'])]
                        exclude_indices.extend(exclude.index.tolist())
    
        return df.drop(exclude_indices, axis=0)
     
    def target_encode(self, train, test, col, smoothing=10):
        """
        Target encoding with additive smoothing for regularization.
        - smoothing=10: Good default (tune 5-20 based on your data).
        """
    
        # Group stats from TRAIN only
        stats = (pd.DataFrame({col: train[col], 'target': train[TARGET_COLUMN]})
                 .groupby(col)
                 .agg(count=('target', 'size'), mean=('target', 'mean'))
                 .reset_index())
    
        global_mean = train[TARGET_COLUMN].mean()
    
        # Smoothed encoding: (count * mean + smoothing * global) / (count + smoothing)
        stats['smoothed'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    
        # Create mapping dict
        mapping = dict(zip(stats[col], stats['smoothed']))
    
        # Apply to train/test
        train[col + "_target_enc"] = train[col].map(mapping)
        test[col + "_target_enc"] = test[col].map(mapping).fillna(global_mean)  # Unseen → global
    
        # Drop original
        train = train.drop(columns=[col])
        test = test.drop(columns=[col] )
        
        return train, test, mapping, global_mean                 
     
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
    
    def remove_outliers(self, df):
        """
        Removing outliers from the data
        1. If the number of bathrooms exceeds 8, then replace that with its corresponding number of bhk
        2. If the total_sqft exceeds 20000, then remove it
        3. Removes BHK outliers: 
           In the same location, a bigger BHK should not be priced significantly lower than a smaller BHK. 
           If it is, that’s suspicious → drop it
        """
        logging.info("Column transformations done")
        logging.info("Starting outlier removals")
        df["bath"] = df.apply( lambda row: min(row["bath"], row["bhk"]) if row["bath"] > 8 else row["bath"], axis=1 )
        df = df[df["total_sqft"] <= 20000].copy()
        df["price_per_sqft"] = df["price"] * 1e5 / df["total_sqft"]
        df = df[df["price_per_sqft"] <= 25000]
        df = df[df["price_per_sqft"] >= 2000]
        df = df.drop(["price_per_sqft"], axis = 1)
        df = self.remove_bhk_outliers(df)
        return df
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started!!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            # Load train and test data
            train_df = self.read_data(file_path= self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path= self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")
            
            
            # Apply custom transformations in specified sequence
            logging.info("Starting data transformation")
            train_df = self.process_location_feature(train_df)
            logging.info("Location transformation done")
            train_df = self.process_size_feature(train_df)
            logging.info("Size transformation done")
            train_df = self.process_society_feature(train_df)
            logging.info("Society transformation done")
            train_df = self.process_bath_feature(train_df)
            logging.info("Bathroom transformation done")
            train_df = self.process_balcony_feature(train_df)
            logging.info("Balcony transformation done")
            train_df = self.process_availability_feature(train_df)
            logging.info("Availability transformation done")
            train_df = self.process_total_sqft_feature(train_df)
            logging.info("Total sqft transformation done")
            train_df = self.remove_outliers(train_df)
            
            test_df = self.process_location_feature(test_df)
            test_df = self.process_size_feature(test_df)
            test_df = self.process_society_feature(test_df)
            test_df = self.process_bath_feature(test_df)
            test_df = self.process_balcony_feature(test_df)
            test_df = self.process_availability_feature(test_df)
            test_df = self.process_total_sqft_feature(test_df)
            logging.info("Custom transformations applied to train and test data")
            
            logging.info("Start target encoding applied to the train and test data")
            train_df, test_df, mapping, global_mean = self.target_encode(train_df, test_df, 'location')
            logging.info("Got target encoding applied to the train and test data")
            logging.info("Data transformation completed")
            
            os.makedirs(self.data_transformation_config.data_transformation_dir,exist_ok=True)
            
            logging.info(f"Exporting transformed train and test file path.")
            train_df.to_csv(self.data_transformation_config.transformed_train_file_path,index=False,header=True)
            test_df.to_csv(self.data_transformation_config.transformed_test_file_path,index=False,header=True)
            encoding_info = {"mapping": mapping, "global_mean": global_mean}
            save_json_object(encoding_info, self.data_transformation_config.target_encoded_mapping_path)
            save_target_encoded_json_object(encoding_info, "updated_target_mapping/mapitems")
            
            return DataTransformationArtifact(
                transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                target_encoded_mapping_file_path= self.data_transformation_config.target_encoded_mapping_path
            )
            
        except Exception as e:
            raise MyException(e, sys) from e    
        