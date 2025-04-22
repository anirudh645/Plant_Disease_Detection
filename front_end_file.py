import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Path to your trained model
MODEL_PATH = r"D:\Plant_Disease_Detection\Deploy\CNN_plantdiseases_model.keras"

# Load the model once using Streamlit caching
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Image prediction function
def model_predict(image_path):
    H, W, C = 224, 224, 3
    img = cv2.imread(image_path)
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, H, W, C)

    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Class names
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Sidebar
st.sidebar.title("🌱 Plant Disease Detection")
page = st.sidebar.radio("Navigate", ["Home", "Disease Prediction"])

# 🏠 Home Page
if page == "Home":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.markdown("### 🔬 Model Information:")
    st.write("- Model Type: Convolutional Neural Network (CNN)")
    st.write("- Data Set link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data?select=New+Plant+Diseases+Dataset%28Augmented%29")
    st.write(f"- Number of Classes: {len(class_name)}")
    st.write("- Data usage for training: 80% Validation: 20%")
    st.write("- Output: Disease classification based on plant leaf image")
    
    
    if os.path.exists(MODEL_PATH):
        st.success("✅ Model loaded successfully from path.")
    else:
        st.error("❌ Model file not found.")

    st.markdown("---")
    st.info("👉 Please select 'Disease Prediction' from the sidebar to continue.")

# 🔍 Prediction Page
elif page == "Disease Prediction":
    st.header("📷 Upload a Leaf Image to Diagnose Disease")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    save_path = None

    if test_image is not None:
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_container_width=True)  # ✅ Updated here
        else:
            st.warning("Please upload an image first.")

    if st.button("Predict"):
        if save_path:
            with st.spinner("Processing image, please wait..."):
                result_index = model_predict(save_path)
            predicted_label = class_name[result_index]
            st.success(f"🌿 Our Prediction: **{predicted_label}**")
            st.write("Due to the limited training of the model the prediction may not be highly accurate.")
        else:
            st.warning("Please upload an image before predicting.")
