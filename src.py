# Librerias

from cleantext import clean
import codecs
from datetime import datetime
import emoji
from emoji import UNICODE_EMOJI
import itertools
import json

from matplotlib import pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go

from wordcloud import WordCloud,STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
from numpy import array, asarray, zeros
import pandas as pd
import seaborn as sns
import numpy as np
from numpy import asarray
from numpy import zeros
import pickle
import joblib
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import string, re
import tensorflow as tf

import snscrape.modules.twitter as sntwitter
import tweepy
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_columns = None
import pyspark
from pyspark.ml.feature import StopWordsRemover

import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, LabelBinarizer, Binarizer,StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from keras.preprocessing.text import Tokenizer, one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Flatten, GlobalMaxPooling1D, InputLayer, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
py.init_notebook_mode(connected=True)
from sentiment_analysis_spanish import sentiment_analysis

# Funciones

def scraper(lista_p,max_tweets,since,until):
    '''
    Funcion para obtener tweets.
    '''
    
    for p in lista_p:
        data = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(
    '{} since:{} until:{} lang:es'.format(p,since,until)).get_items(), max_tweets))
        return data
    
def remove_usernames_links(tweet):
    '''
    Funcion que elimina los usernames y enlaces de un tweet.
    '''
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('http[^\s]+','',tweet)
    return tweet

def clean_tweet(tweet):
    '''
    Funcion para eliminar con regex la puntuacion, hashtags y simbolos.
    '''
    tweet_lower = tweet.lower()
    tweet_cleaned = re.sub(u"(http\S+)|([“”!?])|([#@])", "", tweet_lower)
    return tweet_cleaned

def polarity_text(df):
    '''
    Funcion para calcular la polaridad de cada tweet.
    '''    
    tweets_links = [remove_usernames_links(tweet) for tweet in tweets_text]

    tweets_text_cleaned = [clean_tweet(tweet) for tweet in tweets_links]
    
    tweets_text_cleaned = [add_space(tweet) for tweet in tweets_text_cleaned]

    polarities = [tweet_polarity(tweet, polarity_dictionary) for tweet in tweets_text_cleaned]

    df_tweets = pd.DataFrame({'Text': tweets_text_cleaned,'Polarity': polarities})
    return df_tweets

def add_space(tweet):
    """
    Funcion para añadir un espacio antes de cada emoji.
    """
    return ''.join(' ' + character + ' ' if is_emoji(character) else character for character in tweet).strip()

def no_emoji(tweet):
    '''
    Funcion para eliminar los emojis de un tweet
    '''
    tweet_no_emoji = clean(tweet,no_emoji=True)
    return tweet_no_emoji

def positive_negative(df):
    '''
    Funcion que etiqueta positivo o negativo segun el score.
    '''
    for item in df:
        if item >= 0:
            Sentiment.append("Positive")
        elif item < 0:
            Sentiment.append("Negative")
            
def sentiment_scores(frase):
    '''
    Funcion que calcula el score de sentimiento
    en español.
    '''
    sentiment = sentiment_analysis.SentimentAnalysisSpanish()
    return sentiment.sentiment(frase)

def preprocess_text(sen):
    '''
    Funcion de limpieza de tags, links,usuarios
    '''
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def remove_tags(text):
    return TAG_RE.sub('', text)


def clean_text(text):
    '''
    Funcion de limpieza de tags, links,usuarios
    '''
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    
    tokens = text.split()
    stop_words = set(stopwords.words('spanish'))
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)
  
def decode_prediction(prediction):
    '''
    Funcion decoding de la puntuacion del modelo
    para que sea 0 o 1.
    '''
    return 'Negative' if prediction < 0.5 else 'Positive'

