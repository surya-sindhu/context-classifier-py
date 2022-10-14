# import pandas as pd
# import numpy as np 
from flask import Flask, render_template, request, redirect , url_for,Response
from transformers import pipeline
classifier1 = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
app = Flask(__name__, static_url_path='/static')
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        candidate_labels_string = request.form['labels']
        if len(candidate_labels_string)==0:
            candidate_labels = ['delay' , 'quality' , 'price','payment','damage','online payment']
        else:
            candidate_labels=candidate_labels_string.split(",")
        sequence_to_classify = request.form['text']
        label=classifier1(sequence_to_classify, candidate_labels)['labels'][0]
        return render_template("index.html", response = label)
    else:


        return render_template('index.html')