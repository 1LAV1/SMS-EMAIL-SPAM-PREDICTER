from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier

import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    # Convert to lowercase
    text = str(text).lower()

    # Tokenize
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    # Join tokens back into a string
    return ' '.join(text)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title(' 📩 EMAIL/SMS Spam Classifier')
custom_css = """
    <style>
    .reportview-container .markdown-text-container {
        max-width: 600px;  /* Adjust the width as needed */
        margin: auto;
    }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
input_sms = st.text_area('Enter the message', height=25)

if st.button('Classify'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])  # Wrap in a list
    # Predict
    predicted = model.predict(vector_input)[0]
    # Display
    if predicted == 1:
        st.header("spam!")
    else:
        st.header("Not Spam ")

