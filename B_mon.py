import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

st.set_page_config(
    page_title="Bahraini Banknote Detection Using Deep Learning",
    page_icon="🇧🇭",
    layout="centered"
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
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .result-box {
        background-color: #f0f9f4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }

    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
        color: #15803d;
    }

    .result-confidence {
        color: #374151;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_currency_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)    
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return ["10BD", "1BD", "20BD", "500 Fils", "5BD"]

try:
    interpreter = load_currency_model()
    class_names = load_class_names()

except Exception as e:
    st.error("Error loading the model or class names.")
    st.write(e)
    st.stop()

def predict_currency(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.uint8:
        scale, zero_point = input_details[0]["quantization"]

        if scale != 0:
            img_array = (img_array / scale) + zero_point

        img_array = img_array.astype(np.uint8)

    else:
        img_array = img_array.astype(input_dtype)

    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)[0]

    output_dtype = output_details[0]["dtype"]

    if output_dtype == np.uint8:
        scale, zero_point = output_details[0]["quantization"]

        if scale != 0:
            preds = (preds.astype(np.float32) - zero_point) * scale

    preds = preds.astype(np.float32)

    pred_idx = np.argmax(preds)
    pred_class = class_names[pred_idx]
    confidence = float(preds[pred_idx])

    return pred_class, confidence, preds

st.markdown(
    '<h1 class="main-title">🇧🇭 Bahraini Banknote Classification</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Upload a photo or use your camera to identify a Bahraini banknote</p>',
    unsafe_allow_html=True
)

tab_upload, tab_camera = st.tabs(["📂 Upload Image", "📸 Use Camera"])

image_source = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image_source = uploaded_file

with tab_camera:
    camera_file = st.camera_input(
        "Take a photo of the banknote"
    )

    if camera_file is not None:
        image_source = camera_file

if image_source is not None:

    image = Image.open(image_source).convert("RGB")

    st.image(
        image,
        caption="Selected image",
        use_container_width=True
    )

    if st.button(
        "🔍 Predict Currency",
        use_container_width=True
    ):

        with st.spinner("Classifying..."):

            pred_class, confidence, preds = predict_currency(image)

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">{pred_class}</div>
                <div class="result-confidence">
                    Confidence: {confidence:.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.subheader("Prediction Breakdown")

        probs_dict = dict(zip(class_names, preds))

        st.bar_chart(probs_dict)

        st.subheader("All Predictions")

        sorted_results = sorted(
            probs_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for currency, probability in sorted_results:
            st.write(
                f"**{currency}:** {probability:.2%}"
            )

st.divider()

st.caption("Bahraini Currency Classification using CNN")
