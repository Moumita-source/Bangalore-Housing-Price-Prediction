import sys
import pandas as pd
import numpy as np
from typing import Optional
import threading

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME, DROP_ID
from src.exception import MyException

class HousingData:
    """
    A class to export MongoDB records as a pandas DataFrame with sampling and alerting.
    """

    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
            self.updated_dataframe = None
        except Exception as e:
            raise MyException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, 
                                       sample_size: Optional[int] = 50000, 
                                       database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports a MongoDB collection as a pandas DataFrame.
        Uses sampling if sample_size is provided to avoid huge memory loads.
        """
        try:
            # Access specified collection
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            print("Fetching data from MongoDB...")

            # Use sampling if sample_size is set
            if sample_size is not None:
                pipeline = [{"$sample": {"size": sample_size}}]
                data = list(collection.aggregate(pipeline))
                print(f"Sample fetched with len: {len(data)}")
            else:
                data = list(collection.find())
                print(f"Full data fetched with len: {len(data)}")

            # Convert to DataFrame
            df = pd.DataFrame(data)
            if DROP_ID in df.columns.to_list():
                df = df.drop(columns=[DROP_ID], axis=1)
            return df

        except Exception as e:
            raise MyException(e, sys)

    def watch_collection(self, collection_name: str, database_name: Optional[str] = None, sample_size: int = 50000):
        
        try:
            print(f"Watching collection '{collection_name}' for changes...")
            
            # Load initial dataset
            self.updated_dataframe = self.export_collection_as_dataframe(
                collection_name= collection_name,
                sample_size= sample_size,
                database_name= database_name
            )
            
            # Start background watcher
            watcher_thread = threading.Thread(
                target = self._watch_changes,
                args = (collection_name, database_name, sample_size),
                daemon= True
            )
            
            watcher_thread.start()
            
        except Exception as e:
            raise MyException(e, sys)   
        
    
    def _watch_changes(self, collection_name, database_name, sample_size):
        
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
                
            print("Waiting for change event (30s timeout)....")
            
            with collection.watch(max_await_time_ms = 30000) as stream:
                try:
                    change = next(stream)
                    
                    if change:
                        print("Change detected!! Reloading dataset...")
                        self.updated_dataframe = self.export_collection_as_dataframe(collection_name, sample_size, database_name)
                        
                except StopIteration:
                    print("No change detected!! Stopping watching thread")
                    
        except Exception as e:
            print(f"Watcher error : {e}")            
                                    
                     