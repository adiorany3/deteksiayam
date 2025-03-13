import os
from keras.models import load_model
from streamlit_extras.add_vertical_space import add_vertical_space
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import datetime
import h5py
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import time

# Define current_year at the module level
current_year = datetime.datetime.now().year

# Page configuration with custom theme
st.set_page_config(
    page_title="Prediksi Penyakit Ayam",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded",
    theme="light"  # You can use "dark" if you prefer dark mode
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(90deg, rgba(240,249,255,1) 0%, rgba(224,242,254,1) 50%, rgba(186,230,253,1) 100%);
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .healthy {
        background: linear-gradient(90deg, rgba(220,252,231,1) 0%, rgba(187,247,208,1) 100%);
        border-left: 6px solid #22c55e;
    }
    .ncd {
        background: linear-gradient(90deg, rgba(254,242,242,1) 0%, rgba(254,226,226,1) 100%);
        border-left: 6px solid #ef4444;
    }
    .coccidiosis {
        background: linear-gradient(90deg, rgba(255,251,235,1) 0%, rgba(254,240,138,1) 100%);
        border-left: 6px solid #eab308;
    }
    .salmonella {
        background: linear-gradient(90deg, rgba(239,246,255,1) 0%, rgba(191,219,254,1) 100%);
        border-left: 6px solid #3b82f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
    }
    .disease-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .confidence-meter {
        height: 10px;
        background-color: #e5e7eb;
        border-radius: 5px;
        margin-top: 10px;
        position: relative;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
    }
    .footer-text {
        text-align: center;
        color: #6b7280;
        padding: 1rem;
        font-size: 0.875rem;
    }
    /* Add these to your existing CSS */
    .result-section {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .gauge-container {
        margin-top: -1rem;
        margin-bottom: -1rem;
    }
    .stExpander {
        border: none !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        border-radius: 8px !important;
        overflow: hidden;
    }
    /* Make text color more visible */
    p, li {
        color: #333333 !important;
    }
    /* Better styling for expandable sections */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1E88E5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animations
chicken_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_umwjhujf.json")
scanning_animation = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_zw0djhar.json")

# Sidebar content with improved styling
with st.sidebar:
    # Add animation
    if chicken_animation:
        st_lottie(chicken_animation, height=180, key="chicken")
    
    st.markdown("<h2 style='text-align: center; color: #1E88E5;'>🤖 Mesin Pendeteksi Penyakit Ayam</h2>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ TENTANG SOFTWARE", expanded=True):
        st.markdown('''
        Software ini dirancang untuk memprediksi penyakit pada unggas berdasarkan visualisasi kotoran ayam menggunakan teknologi vision dan machine learning.
        
        **Deteksi Penyakit:**
        - ✅ **Sehat** - Kondisi normal
        - 🦠 **NCD** - Newcastle Disease
        - 🔬 **Koksidiosis** - Infeksi parasit
        - 🧫 **Salmonela** - Infeksi bakteri
        ''')

    with st.expander("📋 CARA PENGGUNAAN", expanded=False):
        st.markdown('''
        1. 📸 Foto atau upload gambar kotoran ayam dengan pencahayaan yang memadai
        2. 🔄 Software akan memproses dan menganalisis gambar
        3. 📊 Hasil prediksi akan ditampilkan dengan detail penyakit
        
        > ⚠️ Software memberikan prediksi awal, hasil akhir harus dikonfirmasi oleh dokter hewan.
        ''')
    
    with st.expander("🧪 SUMBER DATA", expanded=False):
        st.markdown('''
        Data dikembangkan dengan memanfaatkan database Kaggle yang diproses menggunakan teknologi machine learning. Update data dilakukan secara berkala untuk meningkatkan akurasi.
        
        📊 [Chicken Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/allandclive/chicken-disease-1)
        ''')
    
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>👨‍💻 Developer Contact</p>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><a href='https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/' target='_blank'><img src='https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Logo.svg.original.svg' width='120'></a></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.8rem;'>Galuh Adi Insani</p>", unsafe_allow_html=True)

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
            f.attrs['model_config'] = model_config_string
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

def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 18}},  # Reduced title size
        number = {'font': {'size': 24, 'color': "#1E88E5", 'family': 'Arial Black'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1E88E5" if score > 90 else "#FFC107" if score > 70 else "#F44336"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#FFEBEE'},
                {'range': [50, 80], 'color': '#FFFDE7'},
                {'range': [80, 100], 'color': '#E8F5E9'}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=180,  # Reduced height from 250 to 180
        margin=dict(l=10, r=10, t=30, b=10),  # Reduced margins
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def main():
    # Header
    st.markdown("<h1 class='main-header'>🐔 Sistem Deteksi Penyakit Ayam</h1>", unsafe_allow_html=True)
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Ambil Gambar Kotoran Ayam")
        # Camera input with improved styling
        image = st.camera_input(label="Capture Image", key="First Camera", label_visibility="collapsed")
        
        # Upload option as alternative
        with st.expander("Atau upload gambar"):
            uploaded_image = st.file_uploader("Upload gambar kotoran ayam", type=["jpg", "jpeg", "png"])
            
        # Use either camera or uploaded image
        if uploaded_image is not None and image is None:
            image = uploaded_image
    
    # Initialize model outside the image check to avoid reloading
    model_eval = load_models()
    if model_eval is None:
        st.error("Model tidak dapat dimuat. Silakan periksa file model Anda.")
        return

    with col2:
        if image:
            st.markdown("### 🔍 Hasil Analisis")
            
            # Display the captured image
            with st.expander("Gambar Input", expanded=True):
                st.image(image, caption="Gambar Kotoran Ayam", use_container_width=True)
                
            # Show processing animation
            with st.spinner("Menganalisis gambar..."):
                if scanning_animation:
                    placeholder = st.empty()
                    with placeholder:
                        st_lottie(scanning_animation, height=150, key="scanning")
                time.sleep(1)  # Simulate processing time
                
                # Process the image
                np.set_printoptions(suppress=True)
                class_names = open("labels.txt", "r").readlines()
                
                img = Image.open(image)
                img_array = np.array(img)
                
                img_resized = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)
                img_processed = np.asarray(img_resized).reshape(1, 224, 224, 3)
                img_processed = (img_processed / 127.5) - 1
                
                # Predict using the model
                prediction = model_eval.predict(img_processed)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]
                confidence_percent = int(np.round(confidence_score * 100))
                
                # Clear the animation
                placeholder.empty()
                
                # Display results
                if confidence_score > 0.7:  # Lowered threshold slightly
                    if index == 0:
                        st.markdown("<div class='result-card healthy'>", unsafe_allow_html=True)
                        st.markdown("<p class='disease-title'>✅ Sehat</p>", unsafe_allow_html=True)
                        st.write("Kotoran ayam yang sehat memiliki ciri-ciri tertentu yang mencerminkan kesehatan pencernaan dan kondisi ayam secara keseluruhan.")
                    elif index == 1:
                        st.markdown("<div class='result-card ncd'>", unsafe_allow_html=True)
                        st.markdown("<p class='disease-title'>🦠 Newcastle Disease (NCD)</p>", unsafe_allow_html=True)
                        st.write("Penyakit virus yang sangat menular. Gejala: Gangguan pernapasan, saraf, dan pencernaan. Kotoran bisa berwarna hijau atau kuning, berair, dan mengandung darah. Pencegahan: Vaksinasi rutin.")
                    elif index == 2:
                        st.markdown("<div class='result-card coccidiosis'>", unsafe_allow_html=True)
                        st.markdown("<p class='disease-title'>🔬 Koksidiosis</p>", unsafe_allow_html=True)
                        st.write("Penyakit protozoa yang menyerang usus. Gejala: Kotoran berdarah, diare, penurunan nafsu makan, dan kelemahan. Pencegahan: Menjaga kebersihan kandang, pemberian antikoksidia.")
                    elif index == 3:
                        st.markdown("<div class='result-card salmonella'>", unsafe_allow_html=True)
                        st.markdown("<p class='disease-title'>🧫 Salmonela</p>", unsafe_allow_html=True)
                        st.write("Infeksi bakteri yang dapat menyebabkan gangguan pencernaan. Gejala: Diare, kotoran berwarna hijau atau kuning, penurunan nafsu makan, dan demam. Pencegahan: Menjaga kebersihan kandang dan pakan, pemberian antibiotik jika diperlukan.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display confidence score with gauge chart
                    st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
                    st.plotly_chart(create_gauge_chart(confidence_percent), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Add recommendations based on the disease
                    with st.expander("📋 Rekomendasi Penanganan"):
                        if index == 0:
                            st.markdown("""
                            - ✅ Lanjutkan pemberian pakan berkualitas
                            - ✅ Jaga kebersihan kandang secara rutin
                            - ✅ Pastikan vaksinasi terjadwal dengan baik
                            """)
                        elif index == 1:
                            st.markdown("""
                            - ⚠️ **SEGERA ISOLASI AYAM YANG TERINFEKSI**
                            - 💉 Konsultasikan dengan dokter hewan untuk pengobatan
                            - 🧪 Lakukan desinfeksi menyeluruh pada kandang
                            - 💉 Vaksinasi ulang untuk ayam yang belum terinfeksi
                            """)
                        elif index == 2:
                            st.markdown("""
                            - 💊 Berikan obat anti-koksidiosis sesuai anjuran dokter
                            - 💧 Pastikan ayam mendapatkan cairan yang cukup
                            - 🧼 Bersihkan kandang secara menyeluruh
                            - 🌿 Berikan suplemen untuk memperkuat sistem imun
                            """)
                        elif index == 3:
                            st.markdown("""
                            - 💊 Konsultasikan pemberian antibiotik dengan dokter hewan
                            - 💧 Pastikan hidrasi yang cukup untuk ayam
                            - 🧼 Desinfeksi seluruh area kandang dan peralatan
                            - ⚠️ Perhatikan sanitasi pakan dan air minum
                            """)
                else:
                    st.warning("⚠️ Sesuaikan posisi gambar untuk mendapatkan hasil pembacaan terbaik. Pastikan gambar jelas dan pencahayaan cukup.")
        else:
            # Show instruction when no image is captured
            st.info("👈 Silakan ambil foto atau upload gambar kotoran ayam untuk dianalisis.")
            if scanning_animation:
                st_lottie(scanning_animation, height=200, key="waiting")

    # Footer
    st.markdown("""
    <hr style="height:1px;border:none;color:#333;background-color:#ddd;margin-top:30px;margin-bottom:20px">
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="footer-text">
        <p>© {current_year} Developed by <a href="https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/" target="_blank" 
           style="text-decoration:none; color:#0077B5; font-weight:bold;">Galuh Adi Insani</a> with ❤️</p>
        <p>All rights reserved | Powered by AI & Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hide Streamlit style
    hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()