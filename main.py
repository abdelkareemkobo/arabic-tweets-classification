import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the Arabert model and tokenizer
from transformers import AutoTokenizer, AutoModel

#load your pre_trained model with all its weights
model_name= 'Abdelkareem/arabic_tweets_spam_or_ham'
tokenizer =AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = ["ham", "spam"]

# Tokenize and preprocess the text
def preprocess(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    return inputs

# Predict the label for the given text
def predict_label(text):
    inputs = preprocess(text)
    logits = model(**inputs).logits
    label_id = torch.argmax(logits, dim=1).item()
    return labels[label_id]

# Streamlit app
def main():
    st.title("Arabic Tweets Spam or Ham Classification")
    st.write("Enter a tweet in Arabic to classify it as spam or ham.")

    text = st.text_input("Enter the tweet:")
    
    if st.button("Classify"):
        if text:
            label = predict_label(text)
            st.write(f"Predicted Label: {label}")
        else:
            st.write("Please enter a tweet.")

if __name__ == "__main__":
    main()
