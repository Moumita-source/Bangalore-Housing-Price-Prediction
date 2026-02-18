import sys
from src.entity.config_entity import HousingPredictorConfig
from src.entity.s3_estimator import HousingEstimator
from src.exception import MyException
from src.logger import logging
from src.utils import feature_utils
from pandas import DataFrame


class HousingData:
    def __init__(self,
                area_type,
                availability,
                location,
                size,
                society,
                total_sqft,
                bath,
                balcony
                ):
        """
        Housing Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.area_type = area_type
            self.availability = availability
            self.location = location
            self.size = size
            self.society = society
            self.total_sqft = total_sqft
            self.bath = float(bath) if bath is not None else None
            self.balcony = float(balcony) if balcony is not None else None

        except Exception as e:
            raise MyException(e, sys) from e

    def get_housing_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from HousingData class input
        """
        try:
            
            vehicle_input_dict = self.get_housing_data_as_dict()
            return DataFrame(vehicle_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_housing_data_as_dict(self):
        """
        This function returns a dictionary from HousingData class input
        """
        logging.info("Entered get_housing_data_as_dict method as HousingData class")

        try:
            input_data = {
                "area_type": [self.area_type],
                "availability": [self.availability],
                "location": [self.location],
                "size": [self.size],
                "society": [self.society],
                "total_sqft": [self.total_sqft],
                "bath": [self.bath],
                "balcony": [self.balcony]
            }

            logging.info("Created housing data dict")
            logging.info("Exited get_housing_data_as_dict method as HousingData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class HousingDataPredictor:
    def __init__(self,prediction_pipeline_config: HousingPredictorConfig = HousingPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of HousingPricePredictor
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of HousingDataPredictor class")
            
            dataframe = feature_utils.process_dataframe(dataframe)
            
            model = HousingEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)