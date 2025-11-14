# app.py
import streamlit as st
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, AutoConfig
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
import os

st.set_page_config(page_title="FYP: Text Emotion Detection", layout="centered")

# --- CONFIG: update if your repo id changes ---
HF_REPO_ID = "nisaralikhan1/text-emotion-detection"
TF_MODEL_FILENAME = "tf_model.h5"   # name in HF repo
CACHE_DIR = ".cache_model"

# labels used in your training (same order used when training the model)
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

@st.cache_resource(show_spinner=False)
def load_tokenizer_and_model(hf_repo_id=HF_REPO_ID, tf_filename=TF_MODEL_FILENAME):
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_local_path = hf_hub_download(repo_id=hf_repo_id, filename=tf_filename, cache_dir=CACHE_DIR)
    st.write(f"Model file cached at: `{model_local_path}`")

    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(hf_repo_id, cache_dir=CACHE_DIR)
        st.write("Tokenizer loaded from Hugging Face repo.")
    except Exception as e:
        st.warning("Could not load tokenizer from Hugging Face repo. Please ensure tokenizer files are uploaded.")
        tokenizer = None

    model = None
    try:
        model = TFRobertaForSequenceClassification.from_pretrained(hf_repo_id, from_tf=True, cache_dir=CACHE_DIR)
        st.write("Loaded TF model via transformers.from_pretrained().")
    except Exception:
        try:
            cfg = AutoConfig.from_pretrained(hf_repo_id, cache_dir=CACHE_DIR)
            model = TFRobertaForSequenceClassification.from_config(cfg)
            st.write("Created model architecture from config (no weights).")
            if tokenizer is not None:
                dummy = tokenizer("Hello world", return_tensors="tf", truncation=True, padding="max_length", max_length=128)
                _ = model(**{k: tf.constant(v) for k, v in dummy.items()})
            else:
                seq_len = 128
                dummy_ids = tf.constant([[0] * seq_len], dtype=tf.int32)
                dummy_att = tf.constant([[1] * seq_len], dtype=tf.int32)
                _ = model(input_ids=dummy_ids, attention_mask=dummy_att)
            model.load_weights(model_local_path)
            st.write("Loaded weights into the model from the HDF5 file.")
        except Exception as e:
            st.error("Failed to load model. See logs: " + str(e))
            model = None

    return tokenizer, model

tokenizer, model = load_tokenizer_and_model()

st.title("FYP — Text Emotion Detection")
st.caption("Enter text and click Predict. (Model loaded from Hugging Face.)")

text = st.text_area("Enter text", value="I am very happy today!", height=140)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    elif model is None:
        st.error("Model is not loaded. Check logs above.")
    else:
        if tokenizer is not None:
            inputs = tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=128)
            inputs_tf = {k: tf.constant(v) for k, v in inputs.items()}
            logits = model(**inputs_tf).logits
        else:
            seq_len = 128
            dummy_ids = tf.constant([[0] * seq_len], dtype=tf.int32)
            dummy_att = tf.constant([[1] * seq_len], dtype=tf.int32)
            logits = model(input_ids=dummy_ids, attention_mask=dummy_att).logits

        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = LABELS[pred_idx] if pred_idx < len(LABELS) else str(pred_idx)

        st.write("**Predicted emotion:**", pred_label)
        st.write("**Probabilities:**")
        for i, lab in enumerate(LABELS):
            st.write(f"{lab}: {probs[i]:.4f}")
