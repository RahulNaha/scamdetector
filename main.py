import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


# Function to transform input text
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    y = []
    for i in tokens:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)


# Load vectorizer and model
with open('vectorizer.pkl', 'rb') as v:
    tfidf = pickle.load(v)

with open('model.pkl', 'rb') as m:
    model = pickle.load(m)

# Streamlit app
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip():  # Check if input is not empty
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        # Display prediction
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.warning("Please enter a message to classify.")
