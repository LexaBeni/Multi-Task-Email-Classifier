# Multi-Task Email Classifier

An interactive web application that uses a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model to perform multi-task classification on emails. The model predicts three targets simultaneously from a single email text. This project uses a synthetic dataset generated for educational purposes.

## Model Architecture & Tasks
Instead of running three separate models, this project utilizes a custom **Multi-Task Neural Network** built on top of `bert-base-uncased`. It shares the core BERT embeddings and splits into three separate classification heads:
1. **Category** (Spam, Request, Notification)
2. **Priority** (Low, Medium, High)
3. **Intent** (General Info, Ask for Help, Ask for Action, Warning, Financial, Meeting)

---

## Features
- **Real-time Inference:** Powered by PyTorch and Hugging Face Transformers.
- **Interactive UI:** Built with Streamlit, including quick-sample testing buttons.
- **Confidence Scores:** Displays prediction probabilities using Softmax outputs.
- **Smart Resource Caching:** Uses `@st.cache_resource` to ensure the 400MB+ model is loaded into memory only once.

---

Training Performance
The model was trained using an optimized dataset created by an automated rule-based script (data_generator.py), preventing semantic conflicts and noise.

Spam Accuracy: ~100%

Priority Accuracy: ~100%

Intent Accuracy: ~91% (Convergence reached at epoch 5-7)
