from flask import Flask, request, jsonify
import pandas as pd
from numpy import ndarray
import pickle

app = Flask(__name__)

model = pickle.load(open('RF-DescisionTree_CovidDetectionSymptomsBased.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json(force=True)
    kolom = ['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache', 'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ', 'Gastrointestinal ', 'Abroad travel', 'Contact with COVID Patient', 'Attended Large Gathering', 'Visited Public Exposed Places', 'Family working in Public Exposed Places', 'Wearing Masks', 'Sanitization from Market']
    predicting = []

    for i in kolom:
        predicting.append(request_data[i])
    prediction = [predicting]
    
    X_pred = pd.DataFrame(prediction, index=['Pasien'], columns=kolom)
    pred = ndarray.tolist((model.predict(X_pred)))
    if (pred[0] == 1):
        pred = 'Terkena COVID-19'
    else:
        pred = 'Tidak Terkena COVID-19'
    res = {'Hasil' : pred}
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=True)