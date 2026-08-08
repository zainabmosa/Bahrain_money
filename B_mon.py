import streamlit as st
import numpy as np
from PIL import Image
import json
import os

# TFLite runtime
from tflite_runtime.interpreter import Interpreter


# ==========================================
# Page Settings
# ==========================================

st.set_page_config(
    page_title="Bahraini Currency Classifier",
    page_icon="💵",
    layout="centered",
)


# ==========================================
# File Paths
# ==========================================

MODEL_PATH = "bahraini_currency_model.tflite"
CLASS_NAMES_PATH = "class_names.json"

IMG_SIZE = (224, 224)


# ==========================================
# Custom CSS
# ==========================================

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
    unsafe_allow_html=True,
)


# ==========================================
# Load TFLite Model
# ==========================================

@st.cache_resource
def load_currency_model():

    interpreter = Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    return interpreter


# ==========================================
# Load Class Names
# ==========================================

@st.cache_data
def load_class_names():

    if os.path.exists(CLASS_NAMES_PATH):

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    return [
        "0.5_BHD",
        "1_BHD",
        "5_BHD",
        "10_BHD",
        "20_BHD"
    ]


# ==========================================
# Load Model + Classes
# ==========================================

try:

    interpreter = load_currency_model()
    class_names = load_class_names()

except Exception as e:

    st.error(
        "Error loading the model."
    )

    st.write(e)

    st.stop()


# ==========================================
# Prediction Function
# ==========================================

def predict_currency(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(IMG_SIZE)

    # Convert to NumPy
    img_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    input_dtype = input_details[0]["dtype"]

    # Handle input type
    if input_dtype == np.uint8:

        scale, zero_point = input_details[0]["quantization"]

        img_array = (
            img_array / scale
        ) + zero_point

        img_array = img_array.astype(
            np.uint8
        )

    else:

        img_array = img_array.astype(
            input_dtype
        )

    # Set input
    interpreter.set_tensor(
        input_index,
        img_array
    )

    # Run model
    interpreter.invoke()

    # Get prediction
    preds = interpreter.get_tensor(
        output_index
    )[0]

    # Handle quantized output
    output_dtype = output_details[0]["dtype"]

    if output_dtype == np.uint8:

        scale, zero_point = output_details[0]["quantization"]

        preds = (
            preds.astype(np.float32)
            - zero_point
        ) * scale

    # Get predicted class
    pred_idx = np.argmax(preds)

    pred_class = class_names[
        pred_idx
    ]

    confidence = float(
        preds[pred_idx]
    )

    return (
        pred_class,
        confidence,
        preds
    )


# ==========================================
# Title
# ==========================================

st.markdown(
    '<div class="main-title">💵 Bahraini Currency Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload a photo or use your camera to identify a Bahraini banknote</div>',
    unsafe_allow_html=True
)


# ==========================================
# Upload / Camera Tabs
# ==========================================

tab_upload, tab_camera = st.tabs(
    [
        "📁 Upload Image",
        "📷 Use Camera"
    ]
)

image_source = None


# ==========================================
# Upload Image
# ==========================================

with tab_upload:

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image_source = uploaded_file


# ==========================================
# Camera
# ==========================================

with tab_camera:

    camera_file = st.camera_input(
        "Take a photo of the banknote"
    )

    if camera_file is not None:

        image_source = camera_file


# ==========================================
# Prediction
# ==========================================

if image_source is not None:

    image = Image.open(
        image_source
    ).convert("RGB")

    st.image(
        image,
        caption="Selected image",
        use_container_width=True
    )

    if st.button(
        "🔍 Predict Currency",
        use_container_width=True
    ):

        with st.spinner(
            "Classifying..."
        ):

            (
                pred_class,
                confidence,
                preds
            ) = predict_currency(
                image
            )

        # ==================================
        # Result
        # ==================================

        st.markdown(
            f"""
            <div class="result-box">

                <div class="result-label">
                    {pred_class}
                </div>

                <div class="result-confidence">
                    Confidence: {confidence:.2%}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # ==================================
        # Probability Chart
        # ==================================

        st.subheader(
            "Prediction Breakdown"
        )

        probs_dict = dict(
            zip(
                class_names,
                preds
            )
        )

        st.bar_chart(
            probs_dict
        )
