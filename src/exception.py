import sys
from src.logger import logging

def error_detail_function(error_message, error_detail: sys):
    _,_,exc_tb = error_detail()
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    error_detailed_message = "Error occured in python script name [{0}] in line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error_message)
    )
    
    return error_detailed_message
class CustomException(Exception): 
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_detail_function(error_message, error_detail)
    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        raise CustomException(e,error_detail=sys.exc_info)
## try if the exception works
        