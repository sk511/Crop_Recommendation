import pickle
from flask import  Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import json

app=Flask(__name__)

# Loading the model
model=pickle.load(open('crop_recommend.pkl','rb'))

def convert(a):
    crops=['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas','mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate','banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple','orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    return crops[a]

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    # Extract values from the "data" key
    data_values = [list(data.values())]
    print(data_values)

    # Specify column names (optional)
    columns = ['N', 'P', 'K','temperature','humidity','ph','rainfall']

    # Create a DataFrame
    df = pd.DataFrame(data_values, columns=columns)
    print(df)

    output=model.predict(df)
    print(output[0])
    # return jsonify(output[0])
    return convert(output[0])

if __name__=="__main__":
    app.run(debug=True)