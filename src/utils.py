import os,sys
import pickle
import pandas as pd
import numpy as np 

from src.logger import logging
from src.exception import CustomException

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