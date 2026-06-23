import pandas as pd
import plotly.express as px
import joblib
import torch
from transformers import AutoTokenizer
from src.architecture import MultiTaskEmailModel
import streamlit as st
import time

st.markdown("""
This application uses a multi-task BERT model to predict:

- Email Category
- Priority
- Intent
""")

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = MultiTaskEmailModel()
    
    model_path = r'D:\IT\Python\pytorch\email_classification_project\models\multitask_model.pt'
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def predict_text(text):
    tokens = tokenizer(text, truncation = True, max_length = 64, padding=True, return_tensors = 'pt')

    with torch.no_grad():
        outputs = model(tokens["input_ids"], tokens['attention_mask'])

        spam = torch.argmax(outputs["spam"], dim=1).item()
        spam_probs = torch.softmax(outputs["spam"], dim=1)
        spam_confidence = round(spam_probs[0, spam].item(), 2)

        priority = torch.argmax(outputs["priority"], dim=1).item()
        priority_probs = torch.softmax(outputs['priority'], dim=1)
        priority_confidence = round(priority_probs[0, priority].item(), 2)

        intent = torch.argmax(outputs["intent"], dim=1).item()
        intent_probs = torch.softmax(outputs['intent'], dim=1)
        intent_confidence = round(intent_probs[0, intent].item(), 1)

        return spam, priority, intent, spam_confidence, priority_confidence, intent_confidence
    
spam_map = {
    0: "Spam",
    1: "Request",
    2: "Notification"
}

priority_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}

intent_map = {
    0: "General Info",
    1: "Ask for Help",
    2: "Ask for Action",
    3: "Warning",
    4: "Financial",
    5: "Meeting"
}

st.title("📧 Email Classifier")
if "email" not in st.session_state:
    st.session_state['email'] = ""
samples = st.radio("Samples", ["We need to deploy the new database schema by 5 PM today. Please approve the release configuration.", "Congratulations! You won a $1000 cash prize. Claim your financial reward here: http://claim-reward-now.net", "Security notification: A new sign-in was detected on your production server from a new device."], index=None)
if samples:
    st.session_state.email = samples
email = st.text_area("Enter email text", value=st.session_state.email, height=150)
analyze = st.button("Analyze")   
if analyze:
    with st.spinner("Analyzing email..."):
        time.sleep(2)
    spam, priority, intent, spam_confidence, priority_confidence, intent_confidence = predict_text(email)

    st.success("Analysis complete")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(
            f"Category: {spam_map[spam]}, chance {spam_confidence:.1%}"
        )
    with col2:
        if priority == 2:
            st.error(
                f"Priority: {priority_map[priority]}, chance {priority_confidence:.1%}"
            )
        elif priority == 1:
            st.warning(
                f"Priority: {priority_map[priority]}, chance {priority_confidence:.1%}"
            )
        else:
            st.success(
                f"Priority: {priority_map[priority]}, chance {priority_confidence:.1%}"
            )
    with col3:
        st.write(
            f"Intent: {intent_map[intent]}, chance {intent_confidence:.1%}"
        )




