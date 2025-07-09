




#from tkinter import _test
from sklearn.metrics import accuracy_score

import streamlit as st
import joblib
import string

# Load model and vectorizer
model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Streamlit page settings
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Title
st.title("📰 Fake News Detection App")

# Input
user_input = st.text_area("Paste a news article here:")

# Prediction
if st.button("Check if it's Fake or Real"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Clean and vectorize input
        cleaned = user_input.lower().translate(str.maketrans('', '', string.punctuation))
        vec_input = vectorizer.transform([cleaned])
        prediction = model.predict(vec_input)[0]

        if prediction == "FAKE":
            st.error("🛑 This news article is likely **FAKE**.")
        else:
            st.success("✅ This news article is likely **REAL**.")

