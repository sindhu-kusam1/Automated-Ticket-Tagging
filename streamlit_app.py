import streamlit as st
import pandas as pd
import numpy as np
import dill
import nltk
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

st.title('Automated Ticket Tagging System')

sentence = st.text_area('Please Enter the Query')

button = st.button('Generate Tags')



def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

stopwords= ['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

snow_stemmer = SnowballStemmer(language='english')

if button:
     if sentence == "":
          st.write("Please Enter Text")
     else:
          sentences = [sentence]
          for sentence in tqdm(sentences):
               sentence = re.sub(r"http\S+", "", sentence)
               sentence = BeautifulSoup(sentence, 'lxml').get_text()
               sentence = decontracted(sentence)
               sentence = re.sub("\S*\d\S*", "", sentence).strip()
               sentence = re.sub('[^A-Za-z]+', ' ', sentence)
               sentence = ' '.join([snow_stemmer.stem(word) for word in sentence.split()])
               sentence = ' '.join([i for i in sentence.split() if len(i)>2])
               # https://gist.github.com/sebleier/554280
               sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
          sentences = [sentence]
          with open('train_vectorizer.pickle', 'rb') as file:
               vectorizer = dill.load(file)
          x_test = vectorizer.transform(sentences)
          with open('model_final.pickle', 'rb') as file:
               model = dill.load(file)
          predictions  = model.predict(x_test)
          with open('train_vectorizer_y_labels.pickle', 'rb') as file:
               vectorizer = dill.load(file)
          predictions = vectorizer.inverse_transform(predictions)
          final_predictions = ""
          st.write("Tags :")
          for val in predictions[0]:
               print(val)
               st.write(val)
               st.write("\n")




