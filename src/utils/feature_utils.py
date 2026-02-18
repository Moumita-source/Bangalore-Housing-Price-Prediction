import sys
import pandas as pd
import numpy as np
from src.constants import CONVERSION_FACTORS, LOCATION_THRESHOLD
from src.utils.main_utils import load_target_encoded_json_object
from src.exception import MyException
from src.logger import logging


def extract_bhk(x):
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
        
def convert_total_sqft(x):
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
       
    
def process_location_feature(df):
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
    
def process_size_feature(df):
        """
        Transformations on size feature
        1. Extract the numeric value from the size feature
        2. Fill the missing values in extracted feature with its median
        3. Drop the original size feature
        """
        df["bhk"] = df["size"].apply(extract_bhk)
        median_bhk = df["bhk"].median()
        df["bhk"] = df["bhk"].fillna(median_bhk).astype(int)
        df = df.drop(["size"], axis = 1)
        return df    
    
def process_society_feature(df):
        """
        Transformation on society feature
        1. Extract a binary feature has_society from society feature
        """
        df["has_society"] = df["society"].notnull().astype(int)
        df = df.drop(["society"], axis = 1)
        return df    
    
def process_bath_feature(df):
        """
        Transformation on bath feature
        1. Replace all the missing values of bath feature with that of its corresponding bhk feature
        """
        df.loc[df["bath"].isnull(), "bath"] = df.loc[df["bath"].isnull(), "bhk"]
        return df      
    
def process_balcony_feature(df):
        """
        Transformation on balcony feature
        1. Fill the missing values in balcony with its median
        """
        df["balcony"] = df["balcony"].fillna(df["balcony"].median())
        return df    
    
def process_availability_feature(df):
        """
        Transformation on availability feature
        1. Drop the availability feature as it didnt serve as important feature in modelling later
        """
        df = df.drop("availability", axis = 1)
        return df    
    
def process_total_sqft_feature(df):
        """
        Transformation on total_sqft feature
        1. Extract the numeric values from the total_sqrt with various case conversion into its sqft unit
        """
        df["total_sqft_num"] = df["total_sqft"].apply(convert_total_sqft)
        median_sqft = df["total_sqft_num"].median() 
        df["total_sqft_num"] = df["total_sqft_num"].fillna(median_sqft)
        df = df.drop(columns=["total_sqft"])
        df = df.rename(columns={"total_sqft_num": "total_sqft"})
        return df    
    
def apply_target_encoding(x: pd.DataFrame, mapping_file_path: str, col: str) -> pd.DataFrame:
        """
        Apply target encoding to a given column using a saved mapping + global mean.
    
        Parameters:
        - x: DataFrame to transform
        - mapping_file_path: path to JSON file containing {"mapping": ..., "global_mean": ...}
        - col: column name to encode
    
        Returns:
        - DataFrame with new encoded column added
        """
        loaded_info = load_target_encoded_json_object(mapping_file_path)
        mapping = loaded_info["mapping"]
        global_mean = loaded_info["global_mean"]
        
        x[col + "_target_enc"] = x[col].map(mapping).fillna(global_mean)
        return x    
    
def construct_number_of_bathrooms_per_bhk(df) -> pd.DataFrame:
        """
        Create new feature number of bathrooms per bhk
        """
        df["bath_per_bhk"] = df["bath"] / df["bhk"]
        return df    
    
def construct_total_sqrt_per_bhk(df) -> pd.DataFrame:
        """
        Create number of square feet per bhk
        """
        df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"]
        return df    
    
def construct_extra_bathrooms(df) -> pd.DataFrame:
        """
        Create a flag if number of bathrooms exceeded the number of rooms
        """
        df["extra_bath"] = (df["bath"] > df["bhk"] + 1).astype(int)
        return df    
    
    
def construct_total_sqft_log(df) -> pd.DataFrame:
        """
        Capture non-linear of total-sqft
        """
        df["total_sqft_log"] = np.log1p(df["total_sqft"])
        return df   
    
   
def process_dataframe(x: pd.DataFrame):
        try:
            logging.info("Started processing the dataframe ...")
            x = process_location_feature(x)
            x = process_size_feature(x)
            x = process_society_feature(x)
            x = process_bath_feature(x)
            x = process_balcony_feature(x)
            x = process_availability_feature(x)
            x = process_total_sqft_feature(x)
            x = apply_target_encoding(x, "updated_target_mapping/mapitems.json","location")
            x = construct_number_of_bathrooms_per_bhk(x)
            x = construct_total_sqrt_per_bhk(x)
            x = construct_extra_bathrooms(x)
            x = construct_total_sqft_log(x)
            return x
            
        except Exception as e:
            raise MyException(e, sys) from e            
    