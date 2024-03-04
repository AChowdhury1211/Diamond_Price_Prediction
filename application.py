from src.pipelines.prediction_pipeline import Custom_Data,Predict_Pipeline
from flask import Flask,jsonify,render_template, request
from pymongo.mongo_client import MongoClient
from src.logger import logging
from src.exception import CustomException
import sys

application = Flask(__name__)
app = application

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict", methods = ["GET", "POST"])
def prediction_value():
    try:
        if request.method == "GET":
            return render_template("form.html")
    
        else:
            data = Custom_Data(
                carat=float(request.form.get('carat')),
                depth = float(request.form.get('depth')),
                table = float(request.form.get('table')),
                x = float(request.form.get('x')),
                y = float(request.form.get('y')),
                z = float(request.form.get('z')),
                cut = request.form.get('cut'),
                color= request.form.get('color'),
                clarity = request.form.get('clarity')
            )
            data_frame_new = data.change_to_dataframe()
            predict_pipeline = Predict_Pipeline()
            predicted_data = predict_pipeline.predict(data_frame_new)
        
            result = round(predicted_data[0],2)
            result_dict = {"result" : result}
            logging.info(f"Result obtained{result} and result dictionary created {result_dict}")
        
            data_dict = data.to_dict()
            data_dict.update(result_dict)
            logging.info(f"Data dictionary successfully created: {data_dict}")
            
    
            uri = "mongodb+srv://achowdhury1211:<password>@cluster1.ihgavjm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
            client = MongoClient(uri)
            db = client["Diamond_Information"]
            coll_name = db["Diamond"]
            coll_name.insert_one(data_dict)
            client.close()

            return render_template("form.html", final_outcome = result)
        
    except Exception as e:
        logging.info("Exception occured during API deployment")
        raise CustomException(e,sys)
    
if __name__ == "__main__":
    app.run(host= "0.0.0.0", debug= True)