import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import tensorflow as tf

MODEL_PATH = "bahraini_currency_model.tflite"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

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

image_source = None

tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Use Camera"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image_source = uploaded_file

with tab_camera:
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        image_source = camera_file

if image_source is not None:
    image = Image.open(image_source).convert("RGB")
    st.image(image, caption="Selected image", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.float32:
        img_array = img_array / 255.0
    else:
        img_array = img_array.astype(input_dtype)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    pred_idx = np.argmax(preds)
    pred_class = class_names[pred_idx]
    confidence = preds[pred_idx] * 100

    st.success(f"{pred_class} — Confidence: {confidence:.2f}%")
