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
app.static_folder = 'static'

file_name = 'C:/Users/nites/Desktop/DA Bootcamp Homeworks/Project-3/depression_predictor/dep_model_trained.h5'
loaded_model = load_model(file_name)

@app.route('/', methods=["GET", "POST"])
def landing_page():
    return render_template('index.html')

# prediction function 
def ValuePredictor(to_predict_list):
    print(to_predict_list)
    to_predict_list = np.array(to_predict_list, dtype=np.float32)
    X_scaler = MinMaxScaler().fit(to_predict_list.reshape(34,1))
    to_predict_list_scaled = X_scaler.transform(to_predict_list.reshape(34,1))
    print(to_predict_list_scaled)
    #result = loaded_model.predict_classes([to_predict_list])
    to_predict_list_scaled_reshaped = to_predict_list_scaled.reshape(1,34)
    print(to_predict_list_scaled_reshaped)
    result = loaded_model.predict_classes([to_predict_list_scaled_reshaped])
    print(result) 
    return result[0]

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

