import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Image Classifier", layout="centered")

# App UI Header
st.title("📸 Smart AI Image Classifier")
st.write("Upload an image, and the AI will tell you what's in it!")

# Load the Pre-trained Model (MobileNetV2)
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

model = load_model()

# Image Upload Section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("---")
    st.write("🤖 **AI is thinking...**")

    # Preprocessing the image to fit the model requirements
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Make Predictions
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    # Show Results in a user-friendly way
    st.subheader("Results:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.success(f"{i+1}. **{label.replace('_', ' ').title()}** (Confidence: {score*100:.2f}%)")
        st.progress(float(score))

else:
    st.info("Please upload an image file to start the classification.")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit & MobileNetV2 | No API Required")
