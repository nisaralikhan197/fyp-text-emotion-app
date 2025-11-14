# app.py (PyTorch loading; works without TensorFlow)
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import os

st.set_page_config(page_title="FYP: Text Emotion Detection", layout="centered")

# --- CONFIG ---
HF_REPO_ID = "nisaralikhan1/text-emotion-detection"
TF_MODEL_FILENAME = "tf_model.h5"   # file you uploaded to HF (weights in TF format)
CACHE_DIR = ".cache_model"

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

@st.cache_resource(show_spinner=False)
def load_tokenizer_and_model(repo_id=HF_REPO_ID, tf_filename=TF_MODEL_FILENAME):
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_local_path = hf_hub_download(repo_id=repo_id, filename=tf_filename, cache_dir=CACHE_DIR)
    st.write(f"Model weights cached at `{model_local_path}`")

    # load tokenizer from HF repo
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=CACHE_DIR)
        st.write("Tokenizer loaded from Hugging Face repo.")
    except Exception as e:
        st.error("Could not load tokenizer from HF repo. Please ensure tokenizer files are uploaded.")
        raise

    # Try to load the model as a PyTorch model but using TF weights if needed
    model = None
    try:
        # from_pretrained will auto-detect and convert TF weights if from_tf=True
        # we set from_tf=True to tell HF to treat the stored weights as TF h5
        # device_map=None and map_location ensures CPU use
        model = AutoModelForSequenceClassification.from_pretrained(repo_id, from_tf=True, cache_dir=CACHE_DIR, torch_dtype=torch.float32, low_cpu_mem_usage=False)
        st.write("Loaded model via transformers (PyTorch) using TF weights conversion.")
    except Exception as e:
        st.error("Failed to load model via transformers with from_tf=True: " + str(e))
        raise

    # set to eval and CPU
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    return tokenizer, model, device

tokenizer, model, device = load_tokenizer_and_model()

st.title("FYP — Text Emotion Detection (PyTorch)")
st.caption("Model loaded from Hugging Face and converted to PyTorch in the app (no TensorFlow required).")

text = st.text_area("Enter text", value="I am very happy today!", height=140)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Tokenizing and running model..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            # move tensors to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits.cpu().numpy()[0]
                probs = np.exp(logits) / np.sum(np.exp(logits))
                pred_idx = int(np.argmax(probs))
                pred_label = LABELS[pred_idx] if pred_idx < len(LABELS) else str(pred_idx)

        st.write("**Predicted emotion:**", pred_label)
        st.write("**Probabilities:**")
        for i, lab in enumerate(LABELS):
            st.write(f"{lab}: {probs[i]:.4f}")
