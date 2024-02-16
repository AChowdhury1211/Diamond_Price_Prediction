import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import Evaluate_Model
from dataclasses import dataclass
import sys,os
@dataclass 
class Model_Trainer_Config:
    model_trainer_path = os.path.join("artifacts", "model_trainer.pkl")
class Model_Trainer:
    def __init__(self):
        self.model_trainer_config = Model_Trainer_Config()
        
    def Initiate_Model_Trainer(self, train_arr, test_arr):
        try:
            X_train, X_test, y_train, y_test = (
                train_arr[:,:-1], #independent
                test_arr[:,:-1], #independent
                train_arr[:,-1], #target
                test_arr[:,-1] #target
            )
            
            models = {
                "LinearRegression" : LinearRegression(),
                "Lasso" : Lasso(),
                "Ridge" : Ridge(),
                "ElasticNet" : ElasticNet(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "KNearestNeighbours" : KNeighborsRegressor()
            }
            model_report:dict= Evaluate_Model(X_train,X_test,y_train, y_test, models)
            Report = pd.DataFrame(model_report)
            logging.info("Model Report succesfully created")
            
            metrics = ["mean_squared_error","mean_absolute_error","r2_percentage","root_mean_squared_error"]
            for n in range(len(list(Report.iterrows()))):
                
                Best_score = Report.iloc[n].max()
                Best_Model = Report.columns[Report.eq(Best_score).any()][0]
                print(f"The winner is {Best_Model} with score of: {Best_score} in {metrics[n]}")     
            
            Best_R2_Model = Report.columns[Report.eq(Report.iloc[2].max()).any()][0]
            Saved_model = models[Best_R2_Model]
            print(Best_R2_Model)
            
            logging.info(f"Best Model Found")
            
            save_object(
                file_path= self.model_trainer_config.model_trainer_path,
                obj = Saved_model
            )
            
        except Exception as e:
            logging.info("Exception occured during Model Training")
            raise CustomException(e,sys)