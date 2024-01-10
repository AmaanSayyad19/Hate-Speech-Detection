import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

@st.cache(allow_output_mutation=True)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained('./my_moderation_model')
    tokenizer = AutoTokenizer.from_pretrained('./my_moderation_model')
    return model, tokenizer

model, tokenizer = load_model()

st.title('Hate Speech Detection')

user_input = st.text_area("Enter text here:")

if st.button('Predict'):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    st.write('Hate speech detected' if predicted_class == 1 else 'No hate speech detected')
