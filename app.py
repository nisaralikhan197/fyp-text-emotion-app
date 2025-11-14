# app.py (Streamlit + Hugging Face Inference API)
import streamlit as st
import requests
import numpy as np

st.set_page_config(page_title="FYP: Text Emotion Detection (HF Inference)", layout="centered")

# --- CONFIG (change only if you uploaded model under a different HF username/repo) ---
HF_MODEL = "nisaralikhan1/text-emotion-detection"   # Hugging Face model repo id
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Map model label indices (LABEL_0 ..) to your emotion names (must match training order)
LABEL_MAP = {
    "LABEL_0": "sadness",
    "LABEL_1": "joy",
    "LABEL_2": "love",
    "LABEL_3": "anger",
    "LABEL_4": "fear",
    "LABEL_5": "surprise"
}

st.title("FYP — Text Emotion Detection (Hugging Face Inference)")
st.caption("Model runs on Hugging Face servers; this app calls the HF Inference API.")

text = st.text_area("Enter text to analyze", value="I am very happy today!", height=140)

st.markdown("**Note:** If the model repo is private, add a Hugging Face token under app Settings → Secrets as `HF_TOKEN`.")

def query_hf_inference(payload: dict, token: str | None = None):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except Exception as e:
        # show HF error message if present
        st.error(f"Hugging Face API error: {resp.status_code} — {resp.text}")
        raise
    return resp.json()

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        token = None
        # If user added HF_TOKEN in Streamlit secrets, use it
        if "HF_TOKEN" in st.secrets:
            token = st.secrets["HF_TOKEN"]

        with st.spinner("Sending text to Hugging Face Inference API..."):
            # For sequence classification, HF accepts {"inputs": "text"} and returns list of {label,score}
            try:
                hf_payload = {"inputs": text, "options": {"wait_for_model": True}}
                out = query_hf_inference(hf_payload, token=token)
            except Exception:
                st.stop()

        # HF may return errors or a dict with "error"
        if isinstance(out, dict) and out.get("error"):
            st.error("HF Inference API returned an error: " + out.get("error"))
        else:
            # Expected: [{'label': 'LABEL_1', 'score': 0.8}, ...] OR sometimes {'label':..., 'score':...}
            if isinstance(out, dict):
                # single-label case
                out = [out]

            # Convert to stable format and map labels
            try:
                probs = {}
                for item in out:
                    label = item.get("label")
                    score = float(item.get("score", 0.0))
                    mapped = LABEL_MAP.get(label, label)
                    probs[mapped] = score
            except Exception:
                st.error("Unexpected response from HF API. See raw output below.")
                st.write(out)
            else:
                # Sort by score desc
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                pred_label, pred_score = sorted_probs[0]
                st.write("**Predicted emotion:**", pred_label)
                st.write("**Probabilities:**")
                for lab, sc in sorted_probs:
                    st.write(f"{lab}: {sc:.4f}")
