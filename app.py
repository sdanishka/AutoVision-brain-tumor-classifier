import streamlit as st
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from clerk import Clerk

# Initialize Clerk client
clerk_client = Clerk("sk_test_wBCikw9fWpuchaxuHcUA6Ie6t9wCMNT7XDqmacZRg6")

st.set_page_config(
    layout="wide",
    page_title='Human disease detection',
    page_icon='🦴',
)


def verify_user(user_id):
    try:
        # Fetch user from Clerk
        user = clerk_client.users.get_by_id(user_id)
        return True
    except Exception as e:
        return False


def bone_fracture():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)

        arr = img_to_array(image)
        arr = cv2.resize(arr, (150, 150))
        arr = arr.reshape(-1, 150, 150, 3)
        arr = arr / 255

        prediction = BONE_MODEL.predict([arr])
        confidence_level = round(prediction.max(), 2)
        predicted_class = BONE_FRACTURE_CLASSES[prediction.argmax()]
        st.write(f'Predicted Result : {predicted_class}  and Confidence Level : {confidence_level}')


def main():
    user_id = st.experimental_get_query_params().get("userId", [None])[0]
    if user_id and verify_user(user_id):
        st.title('AutoVision Bone Fracture')
        st.write("Provide quick and accurate predictions bone fractures.")
        bone_fracture()
    else:
        st.warning("Please log in to access this app.")


if __name__ == "__main__":
    main()
