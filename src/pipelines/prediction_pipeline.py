from src.logger import logging
from src.exception import CustomException
import os,sys
from src.utils import load_object
import pandas as pd
from dataclasses import dataclass

@dataclass
class Predict_Pipeline:
    def predict(self,features):
        preprocessor_pickle_file_path = os.path.join("artifacts", "preprocessor.pkl")
        model_pickle_file_path = os.path.join("artifacts", "model_trainer.pkl")
        
        preprocessor_loaded = load_object(preprocessor_pickle_file_path)
        model_loaded = load_object(model_pickle_file_path)
        
        Transformed_data = preprocessor_loaded.transform(features)
        Predicted_output = model_loaded.predict(Transformed_data)
        
        return Predicted_output
    
class Custom_Data:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity
        
    def change_to_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame collected")
            return df
        except Exception as e:
            logging("Exception occured at dataframe creation")
        
    def to_dict(self):
        return {
            'carat': self.carat,
            'depth': self.depth,
            'table': self.table,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'cut': self.cut,
            'color': self.color,
            'clarity': self.clarity
        }