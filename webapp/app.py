import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import os
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load final DenseNet201 model
model = tf.keras.models.load_model('densenet201_best_model_bayes_optimization.h5', compile=False)

# Labels
labels = [
    "adult_rice_weevil",
    "house_centipede",
    "american_house_spider",
    "bedbug",
    "brown_stink_bug",
    "carpenter_ant",
    "cellar_spider",
    "flea",
    "silverfish",
    "subterranean_termite",
    "tick"
]

label_map = {
    "adult_rice_weevil": "Rice Weevil",
    "house_centipede": "House Centipede",
    "american_house_spider": "American House Spider",
    "bedbug": "Bedbug",
    "brown_stink_bug": "Brown Stink Bug",
    "carpenter_ant": "Carpenter Ant",
    "cellar_spider": "Cellar Spider",
    "flea": "Flea",
    "silverfish": "Silverfish",
    "subterranean_termite": "Subterranean Termite",
    "tick": "Tick"
}

def preprocess(image):
    '''
    Param: user uploaded image

    Function: resizes, normalizes, converts RGBA to RGB, & adds batch dimension
    
    Returns: preprocessed image as array
    '''
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # normalize to [0, 1]
    if image.shape[-1] == 4: # If image is RGBA
        image = image[..., :3]  # change to RGB
    image = np.expand_dims(image, axis=0)  # since it's a keras model, need (1, 224, 224, 3) input
    return image

# reference: https://github.com/pytholic/streamlit-image-classification/blob/main/app/app.py
def predict(image):
    '''
    Param:
        image: uploaded user image

    Function: preprocesses image & runs model prediction

    Returns: predictions w their probabilities
    '''
    try:
        # Image must be preprocessed first
        image_array = preprocess(image)

        # Predictions using our final DenseNet201 model, returns array of probabilities for each class
        probabilities = model.predict(image_array)[0]

        # Top 3 probability predictions indices
        top_indices = np.argsort(probabilities)[::-1][:3]

        # Each prediction with its corresponding label
        predictions = [(probabilities[i], label_map.get(labels[i], labels[i])) for i in top_indices]
        return predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return []


def app():
    # Title, subheader, paragraph
    st.title("BugBot")
    st.subheader("A tool to classify those nasty pests in your home. 🏠🪲")
    st.write("""
    This tool was created to help New England residents in identifying common household pests using deep learning techniques. 
    By utilizing a balanced and diverse dataset of insect images, BugBot provides a reliable, accurate, and user-friendly tool 
    that helps users classify insects in their homes quickly and confidently. This eliminates the hassle of misclassifying insects or spending 
    time searching for information on the internet.

    """)

    # File uploader and image display
    uploaded_file = st.file_uploader("Upload an insect image...", type=["jpg", "jpeg", "png"])

    predictions = []

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        predictions = predict(image)

    if predictions:
        st.write("### Top 3 Predictions:")
        for i, (prob, label) in enumerate(predictions):
            st.write(f"{i+1}. {label} ({prob*100:.2f}%)")
            st.progress(float(prob))
    else:
        st.write("No predictions.")

if __name__ == "__main__":
    app()
