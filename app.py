#!/usr/bin/python3
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
import joblib
from sklearn.linear_model import LogisticRegression

####################
#   How to use:
#   run python app.py
#   open browser: 
#   http://127.0.0.1:5000/result   (Total_records: This tells you how many records are in the test_x sets )
#   http://127.0.0.1:5000/details/3  (Checking x_values: this tells you the selected 'x2,x6' value for test_x[3]) 
#   http://127.0.0.1:5000/prediction/12 (Checking predicted_values: this tells you the predicted y value for test_x[12])
##################

app = Flask(__name__)
api = Api(app)

######  load models
lr = joblib.load("model_LR.pkl") 
print ('Model loaded')
x_test= joblib.load("x_test.pkl") 
print ('x test values loaded')

class Cellphones(Resource):
    def get(self):
        result = x_test.shape[0]
        return {'Total cellphone records': result} 

class Details(Resource):
    def get(self, index):
        values = x_test[int(index)]
        result =  {'x2, x6': [values.item(0),values.item(1)]}
        return jsonify(result)

class Prediction(Resource):
    def get(self, index):
        values = x_test[int(index)]
        y_pre = lr.predict(values)
        #print (y_pre)
        result = {'line': index, 'prediction':y_pre.item(0)}
        return jsonify(result)


api.add_resource(Cellphones, '/result') 
api.add_resource(Details, '/details/<index>') 
api.add_resource(Prediction, '/prediction/<index>') 



if __name__ == '__main__':
     app.run()
