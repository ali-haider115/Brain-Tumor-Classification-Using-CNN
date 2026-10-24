import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
MODEL_PATH = "models/brain_tumor_model.keras"
TRAIN_DIR = "dataset/train"
IMAGE_SIZE = (128, 128)

# Page Configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Caching Functions (Optimized Loading)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_keras_model():
    model_path = hf_hub_download(
        repo_id="alihaider115/brain-tumor-cnn-model",
        filename="brain_tumor_model.keras"
    )
    return tf.keras.models.load_model(model_path)

@st.cache_data
def get_class_names():
    """Get class names from dataset directory or fallback defaults."""
    if os.path.exists(TRAIN_DIR):
        return sorted(os.listdir(TRAIN_DIR))
    # Fallback default classes if dataset folder is not present in deployment
    return ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
st.title("Brain Tumor Classification")
st.markdown(
    "Upload a Brain MRI image to analyze and classify it using the trained Deep Learning Model."
)

# Load Model & Classes
model = load_keras_model()
class_names = get_class_names()

if model is None:
    st.error(f"❌ Model file not found at `{MODEL_PATH}`. Please run `train.py` first to generate the model.")
    st.stop()

# Sidebar Info
st.sidebar.header("📌 Project Details")
st.sidebar.write("**Input Size:** 128 x 128 x 3")
st.sidebar.write("**Model Architecture:** CNN (Keras)")
st.sidebar.write("**Classes:**", ", ".join(class_names))

# File Uploader
uploaded_file = st.file_uploader(
    "Choose an MRI image file...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    # Display Uploaded Image
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

    # Inference & Results
    with col2:
        st.subheader("Classification Result")
        
        with st.spinner("Analyzing MRI image..."):
            # Preprocess image
            img_resized = image.resize(IMAGE_SIZE)
            img_array = tf.keras.utils.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

            # Model prediction
            predictions = model.predict(img_array, verbose=0)[0]
            predicted_index = np.argmax(predictions)
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        # High-level Metrics Display
        st.success(f"**Predicted Class:** {predicted_class}")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        # Class Probabilities Breakdown
        st.write("### Probability Breakdown")
        prob_df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": np.round(predictions * 100, 2)
        }).sort_values(by="Probability (%)", ascending=False)

        st.bar_chart(prob_df.set_index("Class"))
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

else:
    st.info("💡 Please upload an MRI image (JPG, JPEG, PNG) from the uploader above to view predictions.")