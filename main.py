from keras.models import load_model  # TensorFlow is required for Keras to work
from streamlit_extras.add_vertical_space import add_vertical_space
import cv2  # Install opencv-python
import numpy as np
import streamlit as st
from PIL import Image
import datetime

st.set_page_config(page_title="Prediksi penyakit ayam", page_icon="🐔")

# Sidebar contents
with st.sidebar:
    st.title('🤖 Item Detection')
    st.markdown('''
    ## About
    This app uses a kersa model to detect items using:
    - [Streamlit](https://streamlit.io/)
    - [Teachable Machine](https://teachablemachine.withgoogle.com/)
    ''')
    add_vertical_space(3)
    st.write('This is my linkedin🤗: (https://www.linkedin.com/in/tanviralamsyed/)')
 


    
def main():
     np.set_printoptions(suppress=True)
     image = st.camera_input(label ="Capture Image", key="First Camera", label_visibility="hidden")# this captures the image 
     if image:
        np.set_printoptions(suppress=True)
        model = load_model("keras_Model.h5", compile=False) # this section loads the  model and labels that are used
        class_names = open("labels.txt", "r").readlines()
        
        img = Image.open(image) # stores the natural image so that itcan be manipulated
        img_array = np.array(img)
        
        image = cv2.resize(img_array, (224, 224),interpolation=cv2.INTER_AREA)#resizes it so that it is appropiate for the model to consume theimage
        # st.image(image)
        image = np.asarray(image).reshape(1,224, 224, 3)
        image = (image / 127.5) - 1
        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        print(class_name)
        print(confidence_score)
        if confidence_score > 0.9:
            if index == 0:
                st.subheader("Sehat")
                st.write("Kotoran ayam yang sehat memiliki ciri-ciri tertentu yang mencerminkan kesehatan pencernaan dan kondisi ayam secara keseluruhan.")
            elif index == 1:
                st.subheader("NCD")
                st.write("Penyakit virus yang sangat menular. Gejala: Gangguan pernapasan, saraf, dan pencernaan. Kotoran bisa berwarna hijau atau kuning, berair, dan mengandung darah. Pencegahan: Vaksinasi rutin..")
            elif index ==  2:
                st.subheader("Koksidiosis")
                st.write("Penyakit protozoa yang menyerang usus. Gejala: Kotoran berdarah, diare, penurunan nafsu makan, dan kelemahan. Pencegahan: Menjaga kebersihan kandang, pemberian antikoksidia.")
            elif index == 3:
                st.subheader("Salmonela")
                st.write("Infeksi bakteri yang dapat menyebabkan gangguan pencernaan. Gejala: Diare, kotoran berwarna hijau atau kuning, penurunan nafsu makan, dan demam. Pencegahan: Menjaga kebersihan kandang dan pakan, pemberian antibiotik jika diperlukan.")
            print("Class:", class_name[2:], end="")
            st.write("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        else:
            st.warning("Sesuaikan posisi gambar, untuk mendapatkan hasil pembacaan terbaik")


if __name__ == '__main__':
    current_year = datetime.datetime.now().year
    main()

    # Footer
    st.markdown(f"""
    <div style="text-align: center; padding-top: 20px;">
        © {current_year} Developed by: Galuh Adi Insani. All rights reserved.
    </div>
    """, unsafe_allow_html=True)