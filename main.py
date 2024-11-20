# Install necessary libraries
import streamlit as st
from transformers import pipeline
from PIL import Image

# Function to load the text classification pipeline
@st.cache_resource
def load_text_pipeline():
    return pipeline("text-classification", model="winterForestStump/Roberta-fake-news-detector")

# Function to load the image classification pipeline
@st.cache_resource
def load_image_pipeline():
    return pipeline("image-classification", model="NYUAD-ComNets/NYUAD_AI-generated_images_detector")

# Initialize pipelines
pipe_text = load_text_pipeline()
pipe_img = load_image_pipeline()

# Function to check if text is "fake" or "real"
def check_text(text):
    result = pipe_text(text)
    return result[0]['label']

# Function to check if image is "fake" or "real"
def check_image(image):
    # Open the image using PIL
    image = Image.open(image).convert("RGB")
    # Perform inference using the preloaded pipeline
    result = pipe_img(image)
    # Get the label with the highest score
    highest_result = max(result, key=lambda x: x['score'])
    highest_label = highest_result['label']
    highest_score = highest_result['score']
    # Return the result as a formatted string
    return f"The image is most likely {highest_label} with a score of {highest_score:.2f}"

# Streamlit app
def main():
    st.title("Fake or Real Checker")
    st.write("Enter text or upload an image to check if it's fake or real. Only one at a time.")

    # Input fields
    text = st.text_area("Enter Text (leave blank if uploading an image)")
    image_file = st.file_uploader("Upload Image (not displayed)", type=["png", "jpg", "jpeg"])

    # Check button
    if st.button("Check"):
        if text and not image_file:
            with st.spinner("Analyzing text..."):
                result = check_text(text)
            st.success(f"Result: {result}")
        elif image_file and not text:
            with st.spinner("Analyzing image..."):
                result = check_image(image_file)
            st.success(f"Result: {result}")
        elif text and image_file:
            st.warning("Please enter text or upload an image, not both.")
        else:
            st.warning("Please enter text or upload an image.")

if __name__ == "__main__":
    main()
