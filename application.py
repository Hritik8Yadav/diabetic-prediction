from flask import Flask, request, app, render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

scaler=pickle.load(open('model/scaler_diabitic.pkl','rb'))
regress=pickle.load(open('model/regression.pkl','rb'))


application = Flask(__name__)
app = application


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=''
    if request.method=='POST':

        pregnancy=int(request.form.get('Pregnancies'))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=float(request.form.get('Age'))

        data = scaler.transform([[pregnancy,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict = regress.predict(data)

        if predict == 1:
            result = 'Diabetic'
        else:
            result = 'Non Diabetic'

        return render_template('single_prediction.html',result=result)
    
    else:
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")
