import os,sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Model_Trainer


if __name__ == '__main__':
    obj1 = DataIngestion()
    train_data_path , test_data_path = obj1.initiate_data_ingestion()
    
    print(train_data_path, test_data_path)
    
    obj2 = DataTransformation()
    train_arr, test_arr, preprocessor_path = obj2.initiate_data_transformation(train_data_path, test_data_path)
    
    obj3 = Model_Trainer()
    obj3.Initiate_Model_Trainer(train_arr,test_arr)