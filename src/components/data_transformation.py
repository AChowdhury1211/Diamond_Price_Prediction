from sklearn.impute import SimpleImputer  # handling missing values
from sklearn.preprocessing import StandardScaler  # feature scaling
from sklearn.preprocessing import OrdinalEncoder  # ordinal encoding
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
import os
import sys
import numpy as np
import pandas as pd

from src.utils import save_object

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation initiated")

            categorical_columns = ['cut', 'color', 'clarity']
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1','VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info("Data Transformation pipeline initiated")

            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("OrdinalEncoder", OrdinalEncoder(categories=[
                     cut_categories, color_categories, clarity_categories])),
                    ("Scaler", StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            logging.info("Data Transformation completed")

            return preprocessor

        except Exception as e:
            logging.info(
                "Exception occured data transformation get object stage")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Train and test reading completed")
            logging.info(f"train_df head: \n{train_df.head().to_string()}")
            logging.info(f"test_df head: \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformation_obj()

            target_column = "price"
            drop_column = [target_column, "id"]

            independent_features_train_df = train_df.drop(columns=drop_column, axis=1)
            target_feature_train_df = train_df[target_column]

            independent_features_test_df = test_df.drop(columns=drop_column, axis=1)
            target_feature_test_df = test_df[target_column]

            independent_features_train_arr = preprocessor_obj.fit_transform(independent_features_train_df)
            independent_features_test_arr = preprocessor_obj.transform(independent_features_test_df)

            logging.info("Preprocessing done on train and test dataframes")

            train_arr = np.c_[independent_features_train_arr, np.array(target_feature_train_df)]  
            test_arr = np.c_[independent_features_test_arr, np.array(target_feature_test_df)]  

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            logging.info("Exception at data transformaion initiation")
            raise CustomException(e, sys)
