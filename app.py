from flask import Flask, render_template
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import string
import time
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from scipy.signal import savgol_filter
from scipy import stats
from scipy import ndimage
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import pipeline

app = Flask(__name__)

model = pipeline('sentiment-analysis')

@app.route('/classify', methods=['POST'])
def classify():
    # Get the text from the POST request
    text = request.json['text']

    # Use the pre-trained model to classify the text
    result = model(text)[0]

    # Return a JSON response with the classification result
    return jsonify({'result': result['label']})

if __name__ == '__main__':
    app.run(debug=True)
