import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# parameters
MAX_LENGTH = 200

# load model
model = tf.keras.models.load_model("model/final_lstm_model.keras")

# load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# text cleaning
def clean_text(text):

    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text


# prediction function
def predict_sentiment(review):

    review = clean_text(review)

    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH)

    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        return "Positive 😊", prediction
    else:
        return "Negative 😞", prediction


# UI
st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter a movie review:")

if st.button("Predict"):

    if review.strip() != "":
        result, prob = predict_sentiment(review)

        st.subheader(f"Sentiment: {result}")
        st.write(f"Confidence: {prob:.2f}")

    else:
        st.warning("Please enter a review!")