import streamlit as st
from PIL import Image
import numpy as np
from skin import predict_image


st.title(body = 'Skin Condition AI')

st.sidebar.title('Skin-AI SideBar')
st.sidebar.divider()
decision = st.sidebar.radio(label = 'Choose an Input type', options = ['Upload', 'Camera'])
st.sidebar.divider()
img = None
if decision == 'Camera':
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        # To read image file buffer as a PIL Image:
        img = Image.open(img_file_buffer)
        st.image(img)
else:
    uploaded_file = st.file_uploader("Choose an image file", type="jpg")
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img)

if img is None:
    st.success('Welcome, please upload or snap your picture to start')
else:
    st.divider()
    action = st.button(label = "Analyze Skin")
    if action:
        img.save("temp_img2.jpg")
        prob, preds = predict_image(image_path="temp_img2.jpg")
        st.success(f"The predicted skin condition is: {preds} with probability of: {round(prob,3)}")


