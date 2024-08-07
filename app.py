import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np

# Load your Keras model
model = keras.models.load_model('saved_models/trained.h5')

def make_predictions(image):
    # Resize the image for the model
    img_d = image.resize((244, 244))

    # Check if the image is RGB or not
    if len(np.array(img_d).shape) < 3:
        rgb_img = Image.new("RGB", img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img = img_d

    # Convert the image into a numpy array and reshape
    rgb_img = np.array(rgb_img, dtype=np.float64)
    rgb_img = rgb_img.reshape(1, 244, 244, 3)

    # Make predictions
    predictions = model.predict(rgb_img)
    confidence = np.max(predictions)
    a = int(np.argmax(predictions))
    
    # Set a confidence threshold
    confidence_threshold = 0.5  # You can adjust this threshold

    if confidence < confidence_threshold:
        return "The uploaded image might not be a valid brain X-ray."
    
    if a == 1:
        result = "Result: Glioma Tumor"
    elif a == 2:
        result = "Result: Meningioma Tumor"
    elif a == 3:
        result = "Result: No Tumor"
    else:
        result = "Result: Pituitary Tumor"
    
    return result

# Streamlit UI
st.title("Brain Tumor Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("")

    # Predict button
    if st.button("Predict"):
        predictions = make_predictions(image)
        st.write(predictions)
