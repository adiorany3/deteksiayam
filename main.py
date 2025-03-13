import os
from keras.models import load_model
from streamlit_extras.add_vertical_space import add_vertical_space
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import datetime
import h5py  # Import h5py

st.set_page_config(page_title="Prediksi penyakit ayam", page_icon="🐔")

# Sidebar contents
with st.sidebar:
    st.title('🤖 Mesin pendeteksi penyakit ayam')
    st.markdown('''
    ## Tentang Software
    Software ini diolah dan dirancang untuk mempermudah melakukan prediksi penyakit pada unggas dengan hanya visualisasi kotoran ayam dan menggunakan teknologi vision dan machine learning, software akan menentukan apakah ayam:
    - Sehat
    - NCD
    - Koksidiosis
    - Salmonela

    ## Cara Penggunaan
    1. Foto atau upload gambar kotoran ayam, pastikan dengan pencahayaan yang memadai
    2. Software akan menentukan penyakit ayam dan menampilkan hasil prediksi penyakit ayam
    3. software ini dikembangkan dengan memanfaatkan database Kaggle yang kemudian dilakukan proses training model menggunakan teknologi machine learning, update data akan dilakukan secara berkala untuk mendapatkan hasil yang lebih baik.

    Software memberikan prediksi awal dari hasil visualisasi kotoran ayam, namun hasil akhir harus dikonfirmasi oleh dokter hewan.
    ''')
add_vertical_space(3)
st.markdown("### Connect With Developer")
st.markdown("🔗 [LinkedIn - Galuh Adi Insani](https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/)")
st.markdown("### Data Source")
st.markdown("📊 [Chicken Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/allandclive/chicken-disease-1)")

@st.cache_resource
def load_models():
    model_path = "keras_model.h5"  # Define model path

    # Hack to change model config
    try:
        f = h5py.File(model_path, mode="r+")
        model_config_string = f.attrs.get("model_config")
        if model_config_string is not None and isinstance(model_config_string, bytes):
            model_config_string = model_config_string.decode('utf-8')
        if (model_config_string and '"groups": 1,' in model_config_string):
            model_config_string = model_config_string.replace('"groups": 1,', '')
            f.attrs['model_config'] = model_config_string  # Corrected assignment
            f.flush()
            model_config_string = f.attrs.get("model_config")
            if model_config_string is not None and isinstance(model_config_string, bytes):
                model_config_string = model_config_string.decode('utf-8')
            assert '"groups": 1,' not in model_config_string
        f.close()
    except Exception as e:
        st.warning(f"Error applying model config hack: {e}")

    try:
        model_eval = load_model(model_path, compile=False)
        return model_eval
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def main():
    np.set_printoptions(suppress=True)
    image = st.camera_input(label ="Capture Image", key="First Camera", label_visibility="hidden")
    model_eval = load_models()
    if model_eval is None:
        return

    if image:
        np.set_printoptions(suppress=True)
        class_names = open("labels.txt", "r").readlines()
        
        img = Image.open(image)
        img_array = np.array(img)
        
        image = cv2.resize(img_array, (224, 224),interpolation=cv2.INTER_AREA)
        image = np.asarray(image).reshape(1,224, 224, 3)
        image = (image / 127.5) - 1
        # Predicts the model
        prediction = model_eval.predict(image)
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
            elif index == 2:
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

# Footer with LinkedIn profile link and improved styling
st.markdown("""
<hr style="height:1px;border:none;color:#333;background-color:#333;margin-top:30px;margin-bottom:20px">
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; padding:15px; margin-top:10px; margin-bottom:20px">
    <p style="font-size:16px; color:#555">
        © {current_year} Developed by: 
        <a href="https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/" target="_blank" 
           style="text-decoration:none; color:#0077B5; font-weight:bold">
            <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" 
                 width="16" height="16" style="vertical-align:middle; margin-right:5px">
            Galuh Adi Insani
        </a> 
        with <span style="color:#e25555">❤️</span>
    </p>
    <p style="font-size:12px; color:#777">All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)