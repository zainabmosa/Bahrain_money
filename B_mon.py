import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ---------------------------------
# Page Settings
# ---------------------------------

st.set_page_config(
    page_title="Bahraini Currency Recognition",
    page_icon="💵",
    layout="centered"
)


# ---------------------------------
# File Paths
# ---------------------------------

MODEL_PATH = Path("bahraini_currency_model.keras")
CLASS_NAMES_PATH = Path("class_names.json")

IMG_SIZE = (224, 224)


# ---------------------------------
# Load Model
# ---------------------------------

@st.cache_resource
def load_currency_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "bahraini_currency_model.keras was not found."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ---------------------------------
# Load Class Names
# ---------------------------------

@st.cache_data
def load_class_names():

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "class_names.json was not found."
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    return class_names


# ---------------------------------
# Try Loading Files
# ---------------------------------

try:

    model = load_currency_model()
    class_names = load_class_names()

except Exception as error:

    st.error(
        "Could not load the model or class names."
    )

    st.code(
        str(error)
    )

    st.stop()


# ---------------------------------
# Prediction Function
# ---------------------------------

def predict_currency(image):

    image = image.convert("RGB")

    resized_image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ---------------------------------
# App Title
# ---------------------------------

st.title(
    "💵 Bahraini Currency Recognition"
)

st.write(
    "Upload a photo or use the camera to identify the Bahraini currency."
)


# ---------------------------------
# Choose Image Source
# ---------------------------------

input_method = st.radio(
    "Choose image source:",
    [
        "Upload Image",
        "Use Camera"
    ]
)


selected_file = None


# ---------------------------------
# Upload Image
# ---------------------------------

if input_method == "Upload Image":

    selected_file = st.file_uploader(
        "Upload a currency image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


# ---------------------------------
# Camera
# ---------------------------------

else:

    selected_file = st.camera_input(
        "Take a photo of the currency"
    )


# ---------------------------------
# Show Image and Prediction
# ---------------------------------

if selected_file is not None:

    image = Image.open(
        selected_file
    )

    st.image(
        image,
        caption="Selected Image",
        use_container_width=True
    )

    predict_button = st.button(
        "Predict Currency",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        with st.spinner(
            "Analyzing the image..."
        ):

            (
                predicted_class,
                confidence,
                probabilities
            ) = predict_currency(
                image
            )


        st.success(
            f"Prediction: {predicted_class} BHD"
        )


        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}"
        )


        # ---------------------------------
        # Prediction Probabilities
        # ---------------------------------

        st.subheader(
            "Prediction Probabilities"
        )

        probability_data = {
            class_names[index]:
            float(probabilities[index])

            for index in range(
                len(class_names)
            )
        }


        st.bar_chart(
            probability_data
        )


        # ---------------------------------
        # All Results
        # ---------------------------------

        st.subheader(
            "All Results"
        )

        sorted_results = sorted(
            probability_data.items(),
            key=lambda item: item[1],
            reverse=True
        )

        for class_name, probability in sorted_results:

            st.write(
                f"**{class_name} BHD:** "
                f"{probability:.2%}"
            )


# ---------------------------------
# Footer
# ---------------------------------

st.divider()

st.caption(
    "Deep Learning Project — Bahraini Currency Classification"
)
