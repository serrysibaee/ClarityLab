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
    return "the text is humanly written" if result[0]['label'] == "TRUE" else "the text is synthetically generated"

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
    if highest_label in ["sd", "dalle"]: 
        return f"the image is synthetically generated with a probabilty of {highest_score:.2f}"
    else: 
        return f"The image is real with a probabilty of {highest_score:.2f}"

# Streamlit app
def main():
    st.title("ClarityLab (prototype)")
    st.write("'AntiAI to clear you internet sky'")
    # Display Logo
    try:
        image = Image.open("logo.png")
        st.image(image=image, use_column_width=False)  # Adjust size to fit the container width
    except FileNotFoundError:
        st.error("Logo not found. Please ensure 'logo.png' is in the correct directory.")

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
