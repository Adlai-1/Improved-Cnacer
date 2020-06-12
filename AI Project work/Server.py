from flask import Flask,request,jsonify, render_template
import pickle
import numpy as np
import sklearn
import joblib
app = Flask(__name__)

def getmodel():
    global  model
    model = joblib.load('Breast Analysis')
    print("Model has Loaded!")
    

getmodel()

@app.route('/home')
def running():
    return render_template('View.html')

@app.route('/predict', methods=['POST'] )
def predict():
    features = [float(x) for x in request.form.values()]
    feature = np.array(features)
    result = model.predict([feature])
    if(result=="M"):
        return "The Data sample test Positive for Malignant"
    if(result=="B"):
        return "The Data sample test Positive for Bening"


if __name__ == "__main__":
    app.run(debug=True)