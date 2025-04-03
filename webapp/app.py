# FILE DESCRIPTION: -------------------------------------------------------

# This file runs the Streamlit web application. Users are able to uploadn an
# image of an insect and will classify it using our custom DenseNet201 model.
# The model predicts the insect type and provide the top three predictions.

# --------------------------------------------------------------------------



# ----------- IMPORTS ----------------

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import os
tf.config.set_visible_devices([], 'GPU')



# ----------- SETTINGS ----------------

# GPU and tensorflow settings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



# ----------- CONSTANTS ----------------

IMAGE_SIZE = (224, 224)  
FILE_TYPES = ["jpg", "jpeg", "png"]
MODEL_PATH = "model.h5"

# Load final DenseNet201 model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None


# Label mappings
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
    "bedbug": "Bed Bug",
    "brown_stink_bug": "Brown Stink Bug",
    "carpenter_ant": "Carpenter Ant",
    "cellar_spider": "Cellar Spider",
    "flea": "Flea",
    "silverfish": "Silverfish",
    "subterranean_termite": "Subterranean Termite",
    "tick": "Tick"
}



# ----------- FUNCTIONS ----------------

def preprocess(image):
    '''
    Preprocesses user uploaded image for model prediction

    Param: user uploaded image

    Function: resizes, normalizes, converts RGBA to RGB, & adds batch dimension
    
    Returns: preprocessed image as an array
    '''

    # Error check
    if not isinstance(image, Image.Image):
        raise TypeError("Expected image to be a PIL.Image instance.")
    
    try:
        # Image size
        image = image.resize(IMAGE_SIZE)  

        # Normalize pixel values to [0, 1]
        image = np.array(image) / 255.0  
        
        # Convert image to RGBA to RGB
        if image.shape[-1] == 4: 
            image = image[..., :3]
        
        # Reshape for keras model input (1, 224, 224, 3)
        return np.expand_dims(image, axis=0)
    
    except Exception as e:
        raise ValueError(f"Error in preprocessing image: {e}")


def predict(image):
    '''
    Runs model prediction on the uploaded user image

    Param: uploaded user image

    Function: preprocesses image & runs model prediction

    Returns: list[tuple[float, str]] top 3 predictions  with their probabilities
    
    Reference: https://github.com/pytholic/streamlit-image-classification/blob/main/app/app.py
    '''

    # Error check
    if model is None:
        raise RuntimeError("Model is not loaded. Please place model into webapp folder.")
    
    try:
        # Image must be preprocessed first
        image_array = preprocess(image)

        # Predictions using our final DenseNet201 model, returns array of probabilities for each class
        probabilities = model.predict(image_array)[0]
        
        # Top 3 probability predictions indices
        top_indices = np.argsort(probabilities)[::-1][:3]  # Top 3 indices
        return [(float(probabilities[i]), label_map.get(labels[i], labels[i])) for i in top_indices]
    except Exception as e:
        raise RuntimeError(f"Error in prediction: {e}")


def main():
    """
    Main function that runs the Streamlit web application
    """

    # UI setup: title, subheader, paragraph
    st.title("BugBot")
    st.subheader("A tool to classify those nasty pests in your home. 🏠🪲")
    st.write("""
    This tool was created to help New England residents in identifying common household pests using deep learning techniques. 
    By utilizing a balanced and diverse dataset of insect images, BugBot provides a reliable, accurate, and user-friendly tool 
    that helps users classify insects in their homes quickly and confidently. This eliminates the hassle of misclassifying insects or spending 
    time searching for information on the internet.
    """)

    # 1) Upload image feature, user uploads image based on accepted file types
    uploaded_file = st.file_uploader("Upload an insect image...", type=FILE_TYPES)

    # Stores prediction results
    predictions = []
    
    # 2) Check if user uploaded a file
    if uploaded_file:
        try:
            # Converts image to RGB
            image = Image.open(uploaded_file).convert("RGB")

            # Displays image in Streamlit
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Runs prediction on uploaded image and stores the results
            predictions = predict(image)
            
            if predictions:
                # Prints out top 3 predictions with their probabilities
                st.write("### Top 3 Predictions:")
                for i, (prob, label) in enumerate(predictions):
                    st.write(f"{i+1}. {label} ({prob*100:.2f}%)")
                    st.progress(float(prob))
            else:
                st.write("No predictions available.")
        except Exception as e:
            st.error(f"Error processing image: {e}")


    # --------------------------------------------------------------------------
    # TEST CASE / EXPECTED RESULTS when this script is run:

    # Streamlit running in browser
    
    # time completion: <10 seconds to run the script and Streamlit, 2 seconds to upload and classify image
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    main()
