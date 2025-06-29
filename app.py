import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import requests

# 🚀 Google Drive model URL (Direct Download Format)
MODEL_URL = "https://drive.google.com/file/d/1tpNFXEImBFTqRZMAUOHbfzaHEgpRtgAL/view?usp=drive_link"
MODEL_PATH = "image_classifier_model.h5"

# ✅ Download model from Google Drive (only once)
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "wb") as f:
            f.write(requests.get(MODEL_URL).content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# 🖼️ Streamlit UI
st.title("🐶🐱 Cats vs Dogs Classifier")
st.write("Upload an image and let the model predict whether it's a cat or a dog.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((128, 128))  # match training size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "🐶 It's a Dog!" if prediction > 0.5 else "🐱 It's a Cat!"

    st.subheader("Prediction:")
    st.success(result)
