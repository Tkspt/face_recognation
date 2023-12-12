import base64
import streamlit as st
from PIL import Image

def open_on_console(model, image_path):
    class_name, conf_score = model.predict(image_path)
    print("## {}".format(class_name))
    print("### score: {}%".format(int(conf_score * 1000) / 10))
    

def open_on_web(model, bg_image_path):
    with open(bg_image_path, "rb") as f:
        img_data = f.read()

    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    
    # set title
    st.title('Face Recognition App')

    # set header
    st.header('Please upload an avenger face')

    # upload file
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
    
    # display image
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # classify image
        class_name, conf_score = model.predict(file)

        # write classification
        st.write("## {}".format(class_name))
        st.write("### score: {}%".format(int(conf_score * 1000) / 10))
