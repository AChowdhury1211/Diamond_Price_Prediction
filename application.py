from src.pipelines.prediction_pipeline import Custom_Data,Predict_Pipeline
from flask import Flask,jsonify,render_template, request

application = Flask(__name__)
app = application

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict", methods = ["GET", "POST"])
def prediction_value():
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
        
        return render_template("form.html", final_outcome = result)
    
    
if __name__ == "__main__":
    app.run(host= "0.0.0.0", debug= True)