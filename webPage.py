#!/usr/bin/env python
# coding: utf-8

# In[6]:

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

# from nltk.tokenize import word_tokenize

import time
import numpy as np
import pandas as pd
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
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
# import nltk
# from nltk import word_tokenize
# from nltk.util import ngrams

# In[7]:

st.set_page_config(
    page_title="ate_speech_detection",
    page_icon="@",
    layout="wide",
)

st.title("ate_speech_detection")

@st.cache
def get_data():
    dataset = load_dataset("tweets_hate_speech_detection")
    return dataset
dataset = get_data()

def rem_token(x):
    tokens = word_tokenize(x)


    tokens = list(filter(lambda token: token not in string.punctuation, tokens))

    return tokens

df = pd.DataFrame()
st.markdown("loading Data")
df["tweet"]=dataset["test"]['tweet']  
st.markdown("View of data")
st.dataframe(df.head(10))
# df["tweet"]=train_tweet["tweet"].apply(lambda x : " ".join(rem_token(x)) )
# In[11]:





# ckpt = 'Narrativa/byt5-base-tweet-hate-detection'
# tokenizer = AutoTokenizer.from_pretrained(ckpt)
# model = T5ForConditionalGeneration.from_pretrained(ckpt)
# tokenizer2 = AutoTokenizer.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
# model2 = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")

from transformers import pipeline

neg = 0
pos = 0
sentiment_pipeline = pipeline("sentiment-analysis")
# def classify_tweet(tweet):
#     global pos
#     global neg
#     results = sentiment_pipeline(tweet)
#     if results[0]['label'] == 'NEGATIVE':
#         neg += 1
#     elif results[0]['label'] == 'POSITIVE':
#         pos += 1
#     return results[0]['label'], pos, neg
flask_link = 'http://127.0.0.1:5000/'
def classify_tweet(tweet):
    global pos
    global neg
    response = requests.post('http://localhost:5000/classify', json={'text': tweet})
    result = response.json()['result']
    if result == 'NEGATIVE':
        neg += 1
    elif result == 'POSITIVE':
        pos += 1
    return result, pos, neg

#create two plots. One will show the actual graph whereas the other will show predictions
tab1, tab2 = st.tabs(["Input", "Graph"])
fig1, fig2 = st.columns(2)
import plotly.graph_objs as go

with tab1:
    text_input = st.text_input('Enter some text')
    result, p, n = classify_tweet(text_input)
    st.write(f'The text is {result.lower()}')
with tab2:
    placeholder = st.empty()
    fig = go.Figure()
    st.markdown("BAR Representation")
    for tweets in df['tweet']:
        with fig1:
            with placeholder.container():
                newdf = pd.DataFrame()
                res, p, n = classify_tweet(tweets)
                newdf['label'] = ['positive', 'negative']
                newdf['count'] = [p, n]
                fig.data = []  # Clear existing data in the figure
                fig.add_trace(go.Bar(x=newdf['label'], y=newdf['count'], name='Product sales'))
                fig.update_layout(title='Product sales over time')
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
    # with fig2:
    #     print('none')
    #     st.markdown("Pie representation")
    # #     st.write(selectedModels)
    #     newdf2 = pd.DataFrame()
    #     newdf2['Original'] = atest_y
    #     for i in selectedModels:
    #         currModel = models[i]
    #         dres = regressor(currModel, atrain_x, atrain_y, atest_x, atest_y)         
    #         newdf2[i] = dres
    #     newdf2['date'] = df['date'][-teSize:].values
    # #     st.write(newdf2.columns)
    #     graph2 = px.line(newdf2, x='date', y=newdf2.columns, title='Product sales over time')
    #     st.plotly_chart(graph2, theme="streamlit", use_container_width=True)
        
    #Show the Dataframe






