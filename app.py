import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
import regex


# Try loading your trained model and tokenizer
import os

# model_file = 'new_model2.h5'
# model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_file)
# model = load_model(model_path)

# model = load_model("/content/drive/My Drive/Colab Notebooks/MBA/data/new_model2.h5")

model = load_model("new_model.h5")
from keras.models import load_model


with open("tokenizer (2).pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
# def basic_text_cleaning(line_from_column):
#     # This function takes in a string, not a list or an array for the arg line_from_column
    
#     tokenized_doc = word_tokenize(line_from_column)
    
#     new_review = []
#     for token in tokenized_doc:
#         new_token = regex.sub(u'', token)
#         if not new_token == u'':
#             new_review.append(new_token)
    
#     new_term_vector = []
#     for word in new_review:
#         if not word in stopwords.words('english'):
#             new_term_vector.append(word)
    
#     final_doc = []
#     for word in new_term_vector:
#         final_doc.append(wordnet.lemmatize(word))
    
#     return ' '.join(final_doc)
def basic_text_cleaning(token):
    new_token = regex.sub(r'\W+', '', token)
    return new_token

MAX_SEQUENCE_LENGTH = 2049  # 

def predict(model,i):
    clean_text =[]
    i = basic_text_cleaning(i)
    clean_text.append(i)
    sequences = tokenizer.texts_to_sequences(clean_text)
    data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
    pred = model.predict(data)
    return pred

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

st.title("Fake News Detection")

user_input = st.text_area("Enter a news article to check its authenticity:")

if st.button("Submit"):
    prediction = predict(model, user_input)
    if prediction[0][0] > 0.5:
        st.success("This news article appears to be fake.")
    else:
        st.success("This news article appears to be real.")

