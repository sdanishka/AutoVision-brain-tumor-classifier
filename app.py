import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Pqs2Di4XkKfl9U7avfkn"
)

def infer_image(image):
    # Convert image to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    result = CLIENT.infer(buffer, model_id="brain-tumors-detection/2")
    return result

st.title("Brain Tumor Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Perform inference
    st.write("Classifying...")
    try:
        result = infer_image(image)
        # Display results
        st.write("Results:")
        st.json(result)
    except Exception as e:
        st.error(f"An error occurred: {e}")
