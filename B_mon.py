import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ==========================================
# 1. Page Settings
# ==========================================

st.set_page_config(
    page_title="Bahraini Currency Classifier",
    page_icon="💵",
    layout="centered"
)


# ==========================================
# 2. File Paths
# ==========================================

MODEL_PATH = "bahraini_currency_model.keras"
CLASS_NAMES_PATH = "class_names.json"

IMG_SIZE = (224, 224)


# ==========================================
# 3. Load Model
# ==========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ==========================================
# 4. Load Class Names
# ==========================================

@st.cache_data
def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    return class_names


# ==========================================
# 5. Load Everything
# ==========================================

try:

    model = load_model()
    class_names = load_class_names()

except Exception as e:

    st.error(
        "Error loading the model or class names."
    )

    st.write(e)

    st.stop()


# ==========================================
# 6. Prediction Function
# ==========================================

def predict_currency(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to the same size used during training
    image = image.resize(IMG_SIZE)

    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get predicted class
    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = class_names[
        predicted_index
    ]

    # Get confidence
    confidence = probabilities[
        predicted_index
    ]

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ==========================================
# 7. Title
# ==========================================

st.title(
    "💵 Bahraini Currency Classifier"
)

st.write(
    "Upload an image of a Bahraini currency note "
    "and the CNN model will predict its value."
)


# ==========================================
# 8. Image Upload
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Currency Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==========================================
# 9. Prediction
# ==========================================

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    )

    # Display image
    st.image(
        image,
        caption="Uploaded Currency",
        use_container_width=True
    )

    # Predict button
    if st.button(
        "🔍 Predict Currency",
        use_container_width=True
    ):

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

        # ==================================
        # Prediction Result
        # ==================================

        st.success(
            f"Predicted Currency: {predicted_class} BHD"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )


        # ==================================
        # Probability Results
        # ==================================

        st.subheader(
            "Prediction Probabilities"
        )

        probability_data = {}

        for i in range(
            len(class_names)
        ):

            probability_data[
                class_names[i]
            ] = float(
                probabilities[i]
            )

        st.bar_chart(
            probability_data
        )


        # ==================================
        # Detailed Results
        # ==================================

        st.subheader(
            "All Predictions"
        )

        sorted_results = sorted(
            probability_data.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for currency, probability in sorted_results:

            st.write(
                f"**{currency} BHD:** "
                f"{probability:.2%}"
            )


# ==========================================
# 10. Footer
# ==========================================

st.divider()

st.caption(
    "Bahraini Currency Classification using CNN"
)
