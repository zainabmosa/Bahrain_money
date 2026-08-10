import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

st.set_page_config(
    page_title="Bahraini Currency Classifier",
    page_icon=":dollar:",
    layout="centered",
)

MODEL_PATH = "bahraini_currency_model.tflite"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background-color: #F0F9F4;
        border: 1px solid #BBF7D0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
        color: #15803D;
    }
    .result-confidence {
        color: #374151;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_currency_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_resource
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            return json.load(f)
    return ["0.5_BHD", "1_BHD", "5_BHD", "10_BHD", "20_BHD"]

interpreter = load_currency_model()
class_names = load_class_names()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


st.markdown('<div class="main-title">🇧🇭 Bahraini Currency Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a photo or use your camera to identify a Bahraini banknote</div>', unsafe_allow_html=True)

tab_upload, tab_camera = st.tabs([":file_folder: Upload Image", ":camera: Use Camera"])

image_source = None

with tab_upload:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_source = uploaded_file

with tab_camera:
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        image_source = camera_file

if image_source is not None:
    image = Image.open(image_source).convert("RGB")
    st.image(image, caption="Selected image", use_container_width=True)

    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Classifying..."):
        preds = model.predict(img_array)[0]

    pred_idx = np.argmax(preds)
    pred_class = class_names[pred_idx]
    confidence = preds[pred_idx] * 100

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">{pred_class}</div>
            <div class="result-confidence">Confidence: {confidence:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.caption("Prediction breakdown across all classes")
    probs_dict = dict(zip(class_names, preds))
    st.bar_chart(probs_dict)
