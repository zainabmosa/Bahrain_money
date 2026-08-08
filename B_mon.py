import json
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

MODEL_PATH = "bahraini_currency_model.tflite"
CLASS_NAMES_PATH = "class_names.json"

IMG_SIZE = (224, 224)

# ==========================================
# 3. Load TFLite Model
# ==========================================

@st.cache_resource
def load_model():

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


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

    interpreter, input_details, output_details = load_model()
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

    # Get input information
    input_index = input_details[0]["index"]
    input_dtype = input_details[0]["dtype"]

    # Handle quantized models if needed
    if input_dtype == np.uint8:

        input_scale, input_zero_point = input_details[0]["quantization"]

        image_array = (
            image_array / input_scale
        ) + input_zero_point

        image_array = image_array.astype(
            np.uint8
        )

    else:

        image_array = image_array.astype(
            input_dtype
        )

    # Set input tensor
    interpreter.set_tensor(
        input_index,
        image_array
    )

    # Run prediction
    interpreter.invoke()

    # Get output
    output_index = output_details[0]["index"]

    probabilities = interpreter.get_tensor(
        output_index
    )[0]

    # Handle quantized output if needed
    output_dtype = output_details[0]["dtype"]

    if output_dtype == np.uint8:

        output_scale, output_zero_point = output_details[0]["quantization"]

        probabilities = (
            probabilities.astype(np.float32)
            - output_zero_point
        ) * output_scale

    # Make sure probabilities are float
    probabilities = probabilities.astype(
        np.float32
    )

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
