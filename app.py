from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import  GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app=Flask(__name__)
Swagger(app)

lr = pickle.load(open('Logistic_model.pkl','rb'))
tfidf = pickle.load(open('tfidf_vect.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        review = request.form['review']
        data = [review]
        vect = tfidf.transform(data).toarray()
        my_prediction = lr.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)
    