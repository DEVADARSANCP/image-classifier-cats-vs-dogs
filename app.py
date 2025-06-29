import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('cats_vs_dogs_model.h5')

st.title("🐶🐱 Cats vs Dogs Classifier")
st.write("Upload an image and let the model predict!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    result = "🐶 It's a Dog!" if prediction > 0.5 else "🐱 It's a Cat!"

    st.subheader("Prediction:")
    st.success(result)
