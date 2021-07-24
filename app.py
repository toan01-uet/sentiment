from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from data_prep import TextPreprocess,stopwords

stack_model = pickle.load(open(r'models/stacking_mlxtend.sav', 'rb'))
tfidf = pickle.load(open(r"models/tfidf.pkl", "rb"))


app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html') 

@app.route('/predict', methods=['POST'])

def home():
    text = request.form['input']
    tp = TextPreprocess()
    features = tfidf.transform([tp.preprocess(text, stopwords)])
    
    pred = stack_model.predict_proba(features) [:,1]

    return render_template('after.html', proba = pred)

if __name__ == "__main__":
    app.run(debug=False)