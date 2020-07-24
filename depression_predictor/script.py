import pandas as pd
from numpy.random import seed
from tensorflow.keras.models import load_model
import warnings
warnings.simplefilter('ignore', FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from  flask import Flask,render_template, send_from_directory
from flask import request
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import os
print(os.path.abspath('.'))
app = Flask(__name__)

file_name = 'dep_model_trained.h5'
loaded_model = load_model(file_name)

@app.route('/', methods=["GET", "POST"])
def landing_page():
    return render_template('index.html')

# prediction function 
def ValuePredictor(to_predict_list):
    print(np.array(to_predict_list)) 
    # to_predict = np.array(to_predict_list).shape
    result = loaded_model.predict_classes([to_predict_list]) 
    return result[0]

@app.route('/Project-3/<path:path>')
def send_js(path):
    return send_from_directory('..', path)

@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)		 
        if int(result)== 1: 
            prediction ='You may be diagnosed with depression'
        else: 
            prediction ='No sign of Depression'			
        return render_template("result.html", prediction = prediction) 

if __name__ == "__main__":
    app.run(debug=True)

