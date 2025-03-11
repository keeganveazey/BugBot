# Libraries
import streamlit as st
from PIL import Image
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

def display_uploaded_image(uploaded_file):
    """
    Display uploaded image on Streamlit website

    Param: uploaded_file, the uploaded image file from the user
    Function: displays the uploaded image with a caption
    Returns: void
    """
    st.write("File uploaded.")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width = True)

def load_model():
    model = tf.keras.models.load_model('pest_classifier_cnn.h5')
    return model
with st.spinner('Model is being loaded'):
    model = load_model()

def main():
    
    # Title, subheader, and paragraph
    st.title("BugBot")
    st.subheader("A tool to classify those nasty pests in your home. 🏠🪲")
    st.write("""
    This tool was created to help New England residents in identifying common household pests using deep learning techniques. 
    By utilizing a balanced and diverse dataset of insect images, BugBot provides a reliable, accurate, and user-friendly tool 
    that helps users classify insects in their homes quickly and confidently. This eliminates the hassle of misclassifying insects or spending 
    time searching for information on the internet.

    """)
    st.subheader("")

    # File uploader and image display
    uploaded_file = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        display_uploaded_image(uploaded_file)

if __name__ == "__main__":
    main()
