# 🇧🇭 Bahraini Banknote Detection Using Deep Learning

## 📌 Project Description

This project is an AI-based image classification system that recognizes Bahraini currency notes from an uploaded image.

The system uses a **Deep Learning CNN model** to classify Bahraini banknotes into their different denominations.

## 🎯 Objectives

* Recognize Bahraini currency notes using images.
* Classify the currency into the correct denomination.
* Provide a simple and user-friendly interface.
* Deploy the model as a Streamlit web application.

## 💰 Currency Classes

The model can recognize different Bahraini currency denominations, including:

* 0.5 BHD
* 1 BHD
* 5 BHD
* 10 BHD
* 20 BHD

## 🧠 Machine Learning Model

A **Convolutional Neural Network (CNN)** was trained using images of Bahraini banknotes.

The trained model is saved as:

```text
bahraini_currency_model.keras
```

The class names are stored in:

```text
class_names.json
```

## 🖥️ Application

The project uses **Streamlit** to provide an interactive web interface.

Users can upload an image of a Bahraini banknote, and the application predicts its denomination.

## 📁 Project Structure

```text
Bahraini-Currency-Recognition/
│
├── B_mon.py
├── bahraini_currency_model.keras
├── class_names.json
├── requirements.txt
└── README.md
```

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* CNN
* Streamlit
* NumPy
* PIL
* JSON

## 👥 Team

This project was developed collaboratively by the project team using GitHub for version control and collaboration.

## The link below is the app.
https://bahrain-money.streamlit.app/
