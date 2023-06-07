from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)



@app.route("/")
def home():
    return render_template('index.html')


@app.route('/prediksi', methods=["POST"])
def prediksi():

    file = ""

    algoritma = int(request.form['algo'])

    if (algoritma == 1):
        file = "model_dt.jlb"
    else:
        file = "model_knn.jlb" 
   
    data1 = float(request.form['pal'])
    data2 = float(request.form['ds']) 
    data3 = float(request.form['dbp'])
    data4 = float(request.form['sbp'])
    data5 = float(request.form['gender'])

    model = joblib.load(open(file, "rb"))
    
    arr = np.array([[data1, data2, data3, data5, data4]])
    pred = model.predict(arr)
    
    gangguan = ""
    
    if pred[0] == 0:
        gangguan = "Insomnia"
    elif pred[0] == 1:
        gangguan = "Normal"
    else:
        gangguan = "Sleep Apnea"
        
    return render_template('index.html', prediction = "{}".format(gangguan))


@app.route('/predict', methods=["POST"])
def predict():
    
    data1 = float(request.form['a'])
    data2 = float(request.form['b']) 
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)

    # Mengembalikan hasil prediksi dalam format JSON
    result = {
        'prediction': pred.tolist()  # Convert prediction to a list
    }
    
    return jsonify(result)
    

if __name__ == "__main__":
    app.run(debug=True)