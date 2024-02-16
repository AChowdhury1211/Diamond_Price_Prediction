import os,sys
import pickle
import pandas as pd
import numpy as np 

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try: 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info("Object saved succesfully")
    
    except Exception as e:
        logging.info("Exception at saving object")
        raise CustomException(e,sys)
    
    
def Metrics(original_value, predicted_value):
    mse = mean_squared_error(original_value,predicted_value)
    mae = mean_absolute_error(original_value,predicted_value)
    r2_square = r2_score(original_value,predicted_value) 
    r2_percentage = r2_square *100
    rmse = np.sqrt(mean_squared_error(original_value,predicted_value)) 
    return mse,mae,r2_percentage,rmse

def Evaluate_Model(X_train,X_test,y_train, y_test, models):
    try:
        Evaluation = {}
        for j in range(len(models)):
            model = list(models.values())[j]
            model.fit(X_train,y_train)
            
            y_test_predicted = model.predict(X_test)
            
            mse,mae,r2_percentage,rmse = Metrics(y_test, y_test_predicted)
            metrics = [mse,mae,r2_percentage,rmse]
            Evaluation[list(models.keys())[j]] = []            
            for i in range(len(metrics)):
                Evaluation[list(models.keys())[j]].append(metrics[i])
        return Evaluation
            
            
    except Exception as e :
        logging.info("Exception occured during Model Evaluation in utils")
        raise CustomException(e,sys)