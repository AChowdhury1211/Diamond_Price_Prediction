import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")

        try:
            df = pd.read_csv(os.path.join("notebooks/data", "gemstone.csv"))
            logging.info("Pandas DataFrame is getting generated")

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)
            logging.info("Raw data is getting created")

            train_set, test_set = train_test_split(
                df, test_size=0.30, random_state=42)

            train_set.to_csv(
                self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(
                self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion in completed")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Exception at Data Ingestion stage")
            raise CustomException(e, sys)
