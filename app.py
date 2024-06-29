import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Load the model and other components
with open('model.pkl', 'rb') as file:
    model, tfidf, label_encoder = pickle.load(file)

# Streamlit web interface
st.title('Sentiment Analysis on Social Media Data')

user_input = st.text_area("Enter the text for sentiment analysis:")

if st.button('Analyze'):
    processed_input = preprocess_text(user_input)
    vect_input = tfidf.transform([processed_input])
    prediction = model.predict(vect_input)
    sentiment = label_encoder.inverse_transform(prediction)
    st.write(f'Sentiment: {sentiment[0]}')


