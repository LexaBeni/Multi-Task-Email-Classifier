import pandas as pd
import plotly.express as px
import joblib
import torch
from transformers import AutoTokenizer
from src.architecture import MultiTaskEmailModel
import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = MultiTaskEmailModel()

model.load_state_dict(torch.load('D:\IT\Python\pytorch\email_classification_project\models\multitask_model.pt', map_location = 'cpu'))

model.eval()

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
text = st.text_area("Enter email text")
samples = st.radio("Samples", ["We need to deploy the new database schema by 5 PM today. Please approve the release configuration.", "Congratulations! You won a $1000 cash prize. Claim your financial reward here: http://claim-reward-now.net", "Security notification: A new sign-in was detected on your production server from a new device."], index=None)
analyze = st.button("Analyze")
if samples is not None:
    text = samples
if analyze:
    spam, priority, intent, spam_confidence, priority_confidence, intent_confidence = predict_text(text)

    st.success("Analysis complete")

    st.write(
        f"Category: {spam_map[spam]}, chance {spam_confidence}"
    )
    if priority == 2:
        st.error(
            f"Priority: {priority_map[priority]}, chance {priority_confidence}"
        )
    elif priority == 1:
        st.warning(
            f"Priority: {priority_map[priority]}, chance {priority_confidence}"
        )
    else:
        st.success(
            f"Priority: {priority_map[priority]}, chance {priority_confidence}"
        )
    st.write(
        f"Intent: {intent_map[intent]}, chance {intent_confidence}"
    )




