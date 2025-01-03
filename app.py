import streamlit as st
from transformers import pipeline

# Load the pre-trained model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    return pipeline("sentiment-analysis")

# Initialize the model
model = load_model()

# Streamlit UI
st.title("Sentiment Analysis with Hugging Face")
st.write("Analyze the sentiment of text (Positive or Negative).")

# User input
user_input = st.text_area("Enter your text here:", placeholder="Type something...")

# Button for prediction
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Get predictions
        result = model(user_input)[0]
        # Display results
        st.success(f"**Sentiment:** {result['label']}")
        st.info(f"**Confidence Score:** {result['score']:.2f}")
    else:
        st.warning("Please enter some text to analyze.")
