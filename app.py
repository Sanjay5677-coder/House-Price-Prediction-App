from flask import Flask,render_template,request

import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('NewModel.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    feat = [int(x) for x in request.form.values()]
    feat_t = [np.array(feat)]
    prediction = model.predict(feat_t)
    output = round(prediction[0],2)
    return render_template('index.html',prediction='Charges Prediction : {}'.format(output))
    return render_template('index.html',prediction='Charges Prediction : {}')

if __name__ == '__main__':
 app.run(debug=True)