import json
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Function to load Lottie animation from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animation
lottie_hello = load_lottieurl("https://lottie.host/e4e05d73-3844-4358-a361-065189a0a236/0PXL7pkBcP.json")

# Display Lottie animation with smaller size and centered
st_lottie(lottie_hello, key="Hello", width=200, height=200)



nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

try:
    tf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading pickled files: {e}")

st.title("Email/SMS Spam Detector")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    if input_sms:
        transformed_sms = transform_text(input_sms)
        vector_input = tf.transform([transformed_sms])

        try:
            result = model.predict(vector_input)[0]
            if result == 1:
                st.header("The message/Email is a spam")
            else:
                st.header("The message/Email is not a spam")
        except Exception as e:
            st.error(f"Error predicting: {e}")
    else:
        st.warning("Please enter a message")
