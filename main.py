import os
from keras.models import load_model
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
    initial_sidebar_state="expanded"
)

# Apply custom CSS - with stronger selectors and !important flags
st.markdown("""
<style>
    /* Modern Design System - with stronger selectors */
    :root {
        --primary-color: #4F46E5 !important;
        --primary-light: #E0E7FF !important;
        --success-color: #10B981 !important;
        --warning-color: #F59E0B !important;
        --danger-color: #EF4444 !important;
        --info-color: #3B82F6 !important;
        --dark-text: #1F2937 !important;
        --light-text: #6B7280 !important;
        --card-bg: #FFFFFF !important;
        --background: #F9FAFB !important;
        --border-radius: 12px !important;
    }
    
    /* Override Streamlit's background */
    .stApp {
        background-color: var(--background) !important;
    }
    
    /* Typography */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5,
    div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3, div.stMarkdown h4, div.stMarkdown h5 {
        font-family: 'Inter', sans-serif !important;
        color: var(--dark-text) !important;
    }
    
    /* Main Header - force color and background */
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 2rem !important;
        text-align: center !important;
        padding: 1.5rem !important;
        border-radius: var(--border-radius) !important;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.2) !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Card Styling - with stronger selectors */
    div.modern-card {
        background-color: var(--card-bg) !important;
        border-radius: var(--border-radius) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        border: 1px solid #E5E7EB !important;
    }
    
    div.modern-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Result Cards - with stronger selectors */
    div.result-card {
        padding: 1.5rem !important;
        border-radius: var(--border-radius) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        margin-bottom: 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    div.result-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.12) !important;
    }
    
    div.healthy {
        background: linear-gradient(135deg, rgba(220,252,231,1) 0%, rgba(187,247,208,1) 100%) !important;
        border-left: 6px solid var(--success-color) !important;
    }
    
    div.ncd {
        background: linear-gradient(135deg, rgba(254,242,242,1) 0%, rgba(254,226,226,1) 100%) !important;
        border-left: 6px solid var(--danger-color) !important;
    }
    
    div.coccidiosis {
        background: linear-gradient(135deg, rgba(255,251,235,1) 0%, rgba(254,240,138,1) 100%) !important;
        border-left: 6px solid var(--warning-color) !important;
    }
    
    div.salmonella {
        background: linear-gradient(135deg, rgba(239,246,255,1) 0%, rgba(191,219,254,1) 100%) !important;
        border-left: 6px solid var(--info-color) !important;
    }
    
    /* Disease Typography */
    p.disease-title {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.75rem !important;
        color: var(--dark-text) !important;
    }
    
    /* Button Styling */
    .stButton > button, div.stButton > button {
        width: 100% !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        box-shadow: 0 4px 6px rgba(79, 70, 229, 0.25) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover, div.stButton > button:hover {
        box-shadow: 0 6px 10px rgba(79, 70, 229, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Override for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px !important;
        white-space: pre-wrap !important;
        background-color: #F3F4F6 !important;
        border-radius: 4px 4px 0px 0px !important;
        gap: 1px !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: white !important;
    }
    
    /* Info message styling */
    div.stAlert {
        background-color: #EFF6FF !important;
        color: #1E40AF !important;
        border-left-color: #3B82F6 !important;
    }
    
    /* Warning message styling */
    div.stWarning {
        background-color: #FEF3C7 !important;
        color: #92400E !important;
        border-left-color: #F59E0B !important;
    }
    
    /* Error message styling */  
    div.stError {
        background-color: #FEE2E2 !important;
        color: #B91C1C !important;
        border-left-color: #EF4444 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: var(--primary-color) !important;
        background-color: var(--primary-light) !important;
        border-radius: var(--border-radius) !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Custom info card */
    div.info-card {
        border-radius: var(--border-radius) !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        border-left: 4px solid var(--primary-color) !important;
        background-color: var(--primary-light) !important;
    }
    
    /* Footer text */
    div.footer-text {
        text-align: center !important;
        color: var(--light-text) !important;
        padding: 1rem !important;
        font-size: 0.875rem !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in !important;
    }
    
    /* Fix for sidebar */
    section[data-testid="stSidebar"] {
        background-color: white !important;
        border-right: 1px solid #E5E7EB !important;
    }
    
    /* Custom containers */
    div.custom-container {
        background-color: white !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        margin-bottom: 1.5rem !important;
    }
</style>

<!-- Import Google Fonts -->
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if (r.status_code != 200):
        return None
    return r.json()

# Load Lottie animations
chicken_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_umwjhujf.json")
scanning_animation = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_zw0djhar.json")
upload_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_vvcunvas.json")

# Sidebar content with improved styling - using direct HTML for more control
with st.sidebar:
    # Add animation
    if chicken_animation:
        st_lottie(chicken_animation, height=180, key="chicken")
    
    st.markdown('<div style="background-color: #F3F4F6; padding: 15px; border-radius: 12px; margin-bottom: 20px;"><h2 style="text-align: center; color: #4F46E5; margin: 0;">🤖 Mesin Pendeteksi Penyakit Ayam</h2></div>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ TENTANG SOFTWARE", expanded=True):
        st.markdown('''
        <div style="background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%); padding: 15px; border-radius: 12px; margin-bottom: 15px;">
        Software ini dirancang untuk memprediksi penyakit pada unggas berdasarkan visualisasi kotoran ayam menggunakan teknologi vision dan machine learning.
        </div>
        
        <div style="background-color: white; padding: 15px; border-radius: 12px; border: 1px solid #E5E7EB;">
        <strong>Deteksi Penyakit:</strong>
        <ul style="list-style-type: none; padding-left: 5px; margin-top: 10px;">
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="background-color: #D1FAE5; color: #10B981; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 8px;">✅</span>
                <strong>Sehat</strong> - Kondisi normal
            </li>
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="background-color: #FEE2E2; color: #EF4444; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 8px;">🦠</span>
                <strong>NCD</strong> - Newcastle Disease
            </li>
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="background-color: #FEF3C7; color: #F59E0B; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 8px;">🔬</span>
                <strong>Koksidiosis</strong> - Infeksi parasit
            </li>
            <li style="display: flex; align-items: center;">
                <span style="background-color: #EFF6FF; color: #3B82F6; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 8px;">🧫</span>
                <strong>Salmonela</strong> - Infeksi bakteri
            </li>
        </ul>
        </div>
        ''', unsafe_allow_html=True)

    with st.expander("📋 CARA PENGGUNAAN", expanded=False):
        st.markdown('''
        <div style="background-color: white; padding: 15px; border-radius: 12px; border: 1px solid #E5E7EB;">
        <ol style="padding-left: 20px;">
            <li style="margin-bottom: 10px;">📸 Foto atau upload gambar kotoran ayam dengan pencahayaan yang memadai</li>
            <li style="margin-bottom: 10px;">🔄 Software akan memproses dan menganalisis gambar</li>
            <li style="margin-bottom: 10px;">📊 Hasil prediksi akan ditampilkan dengan detail penyakit</li>
        </ol>
        
        <div style="background-color: #FEF2F2; border-left: 4px solid #EF4444; padding: 12px; border-radius: 6px; margin-top: 10px;">
            <strong>⚠️ Perhatian:</strong> Software memberikan prediksi awal, hasil akhir harus dikonfirmasi oleh dokter hewan.
        </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with st.expander("🧪 SUMBER DATA", expanded=False):
        st.markdown('''
        <div style="background-color: white; padding: 15px; border-radius: 12px; border: 1px solid #E5E7EB;">
        Data dikembangkan dengan memanfaatkan database Kaggle yang diproses menggunakan teknologi machine learning. Update data dilakukan secara berkala untuk meningkatkan akurasi.
        
        <a href="https://www.kaggle.com/datasets/allandclive/chicken-disease-1" target="_blank" style="display: flex; align-items: center; background: #F3F4F6; padding: 10px; border-radius: 8px; text-decoration: none; color: #111827; margin-top: 15px;">
            <span style="background: #4F46E5; color: white; width: 30px; height: 30px; display: inline-flex; align-items: center; justify-content: center; border-radius: 50%; margin-right: 10px;">📊</span>
            <span>Chicken Disease Dataset (Kaggle)</span>
        </a>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div style="background-color: #F3F4F6; padding: 15px; border-radius: 12px; margin-top: 20px;"><p style="text-align: center; margin: 0; font-weight: bold;">👨‍💻 Developer Contact</p></div>', unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; padding: 12px; background-color: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 10px; border: 1px solid #E5E7EB;'><a href='https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/' target='_blank'><img src='https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Logo.svg.original.svg' width='120'></a></div>", unsafe_allow_html=True)
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
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 24, 'family': 'Inter, sans-serif'}},
        delta = {'reference': 90, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4F46E5"},
            'bar': {'color': "#4F46E5" if score > 90 else "#F59E0B" if score > 70 else "#EF4444"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E5E7EB",
            'steps': [
                {'range': [0, 50], 'color': '#FEE2E2'},
                {'range': [50, 80], 'color': '#FEF3C7'},
                {'range': [80, 100], 'color': '#D1FAE5'}
            ],
            'threshold': {
                'line': {'color': "#10B981", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': 'Inter, sans-serif'}
    )
    return fig

def main():
    # Header - using direct HTML with inline styling
    st.markdown('<h1 class="main-header fade-in">🐔 AI Deteksi Penyakit Ayam</h1>', unsafe_allow_html=True)
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Use direct HTML for more styling control
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #4F46E5; margin-bottom: 20px;">📸 Ambil Gambar Kotoran Ayam</h3>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📷 Kamera", "📁 Upload"])
        
        with tab1:
            st.markdown('<div style="background-color: #F9FAFB; padding: 15px; border-radius: 12px; border: 1px solid #E5E7EB;">', unsafe_allow_html=True)
            image = st.camera_input(label="Capture Image", key="First Camera", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab2:
            st.markdown('<div style="background-color: #F9FAFB; padding: 15px; border-radius: 12px; border: 1px solid #E5E7EB;">', unsafe_allow_html=True)
            if upload_animation:
                st_lottie(upload_animation, height=150, key="upload_animation")
            uploaded_image = st.file_uploader("Upload gambar kotoran ayam", 
                                             type=["jpg", "jpeg", "png"], 
                                             help="Format yang didukung: JPG, JPEG, PNG")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Use either camera or uploaded image
        if uploaded_image is not None and image is None:
            image = uploaded_image
            
        st.markdown('<div style="background-color: #F3F4F6; padding: 15px; border-radius: 8px; margin-top: 20px; border: 1px solid #E5E7EB;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #4F46E5; margin-bottom: 10px;">💡 Tips Pengambilan Gambar</h4>', unsafe_allow_html=True)
        st.markdown("""
        - Pastikan pencahayaan cukup terang
        - Ambil gambar dari jarak 20-30 cm
        - Pastikan objek terlihat jelas dan fokus
        - Hindari background yang terlalu ramai
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize model outside the image check to avoid reloading
    model_eval = load_models()
    if model_eval is None:
        st.error("Model tidak dapat dimuat. Silakan periksa file model Anda.")
        return

    with col2:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        if image:
            st.markdown('<h3 style="color: #4F46E5; margin-bottom: 20px;">🔍 Hasil Analisis</h3>', unsafe_allow_html=True)
            
            # Display the captured image
            st.markdown('<div style="border-radius: 12px; overflow: hidden; border: 1px solid #E5E7EB; margin-bottom: 15px;">', unsafe_allow_html=True)
            st.image(image, caption="Gambar Kotoran Ayam", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
                
            # Show processing animation
            with st.spinner("Menganalisis gambar..."):
                if scanning_animation:
                    placeholder = st.empty()
                    with placeholder:
                        st_lottie(scanning_animation, height=150, key="scanning")
                time.sleep(1)  # Simulate processing time
                
                # Process the image
                np.set_printoptions(suppress=True)
                try:
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
                    
                    # Create a stylized result section - using direct HTML
                    st.markdown('<div style="animation: fadeIn 0.5s ease-in;">', unsafe_allow_html=True)
                    
                    # Display results with direct HTML styling
                    if confidence_score > 0.7:  # Lowered threshold slightly
                        if index == 0:
                            st.markdown('<div style="background: linear-gradient(135deg, rgba(220,252,231,1) 0%, rgba(187,247,208,1) 100%); border-left: 6px solid #10B981; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
                            st.markdown('<p style="font-size: 1.5rem; font-weight: 800; margin-bottom: 0.75rem; color: #1F2937;">✅ Sehat</p>', unsafe_allow_html=True)
                            st.write("Kotoran ayam yang sehat memiliki ciri-ciri tertentu yang mencerminkan kesehatan pencernaan dan kondisi ayam secara keseluruhan.")
                        elif index == 1:
                            st.markdown('<div style="background: linear-gradient(135deg, rgba(254,242,242,1) 0%, rgba(254,226,226,1) 100%); border-left: 6px solid #EF4444; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
                            st.markdown('<p style="font-size: 1.5rem; font-weight: 800; margin-bottom: 0.75rem; color: #1F2937;">🦠 Newcastle Disease (NCD)</p>', unsafe_allow_html=True)
                            st.write("Penyakit virus yang sangat menular. Gejala: Gangguan pernapasan, saraf, dan pencernaan. Kotoran bisa berwarna hijau atau kuning, berair, dan mengandung darah. Pencegahan: Vaksinasi rutin.")
                        elif index == 2:
                            st.markdown('<div style="background: linear-gradient(135deg, rgba(255,251,235,1) 0%, rgba(254,240,138,1) 100%); border-left: 6px solid #F59E0B; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
                            st.markdown('<p style="font-size: 1.5rem; font-weight: 800; margin-bottom: 0.75rem; color: #1F2937;">🔬 Koksidiosis</p>', unsafe_allow_html=True)
                            st.write("Penyakit protozoa yang menyerang usus. Gejala: Kotoran berdarah, diare, penurunan nafsu makan, dan kelemahan. Pencegahan: Menjaga kebersihan kandang, pemberian antikoksidia.")
                        elif index == 3:
                            st.markdown('<div style="background: linear-gradient(135deg, rgba(239,246,255,1) 0%, rgba(191,219,254,1) 100%); border-left: 6px solid #3B82F6; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
                            st.markdown('<p style="font-size: 1.5rem; font-weight: 800; margin-bottom: 0.75rem; color: #1F2937;">🧫 Salmonela</p>', unsafe_allow_html=True)
                            st.write("Infeksi bakteri yang dapat menyebabkan gangguan pencernaan. Gejala: Diare, kotoran berwarna hijau atau kuning, penurunan nafsu makan, dan demam. Pencegahan: Menjaga kebersihan kandang dan pakan, pemberian antibiotik jika diperlukan.")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display confidence score with gauge chart
                        st.plotly_chart(create_gauge_chart(confidence_percent), use_container_width=True)
                        
                        # Add recommendations based on the disease
                        with st.expander("📋 Rekomendasi Penanganan"):
                            if index == 0:
                                st.markdown("""
                                <ul style="list-style-type: none; padding-left: 0;">
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #D1FAE5; color: #10B981; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">✓</span>
                                        <span>Lanjutkan pemberian pakan berkualitas</span>
                                    </li>
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #D1FAE5; color: #10B981; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">✓</span>
                                        <span>Jaga kebersihan kandang secara rutin</span>
                                    </li>
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #D1FAE5; color: #10B981; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">✓</span>
                                        <span>Pastikan vaksinasi terjadwal dengan baik</span>
                                    </li>
                                </ul>
                                """, unsafe_allow_html=True)
                            elif index == 1:
                                st.markdown("""
                                <div style="background-color: #FEE2E2; border-radius: 8px; padding: 10px; margin-bottom: 15px;">
                                    <strong style="color: #B91C1C;">⚠️ SEGERA ISOLASI AYAM YANG TERINFEKSI</strong>
                                </div>
                                <ul style="list-style-type: none; padding-left: 0;">
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #FEE2E2; color: #EF4444; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">💉</span>
                                        <span>Konsultasikan dengan dokter hewan untuk pengobatan</span>
                                    </li>
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #FEE2E2; color: #EF4444; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">🧪</span>
                                        <span>Lakukan desinfeksi menyeluruh pada kandang</span>
                                    </li>
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #FEE2E2; color: #EF4444; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">💉</span>
                                        <span>Vaksinasi ulang untuk ayam yang belum terinfeksi</span>
                                    </li>
                                </ul>
                                """, unsafe_allow_html=True)
                            elif index == 2:
                                st.markdown("""
                                <ul style="list-style-type: none; padding-left: 0;">
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #FEF3C7; color: #D97706; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">💊</span>
                                        <span>Berikan obat anti-koksidiosis sesuai anjuran dokter</span>
                                    </li>
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #FEF3C7; color: #D97706; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">💧</span>
                                        <span>Pastikan ayam mendapatkan cairan yang cukup</span>
                                    </li>
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #FEF3C7; color: #D97706; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">🧼</span>
                                        <span>Bersihkan kandang secara menyeluruh</span>
                                    </li>
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #FEF3C7; color: #D97706; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">🌿</span>
                                        <span>Berikan suplemen untuk memperkuat sistem imun</span>
                                    </li>
                                </ul>
                                """, unsafe_allow_html=True)
                            elif index == 3:
                                st.markdown("""
                                <ul style="list-style-type: none; padding-left: 0;">
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #EFF6FF; color: #3B82F6; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">💊</span>
                                        <span>Konsultasikan pemberian antibiotik dengan dokter hewan</span>
                                    </li>
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #EFF6FF; color: #3B82F6; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">💧</span>
                                        <span>Pastikan hidrasi yang cukup untuk ayam</span>
                                    </li>
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #EFF6FF; color: #3B82F6; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">🧼</span>
                                        <span>Desinfeksi seluruh area kandang dan peralatan</span>
                                    </li>
                                    <li style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: #EFF6FF; color: #3B82F6; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">⚠️</span>
                                        <span>Perhatikan sanitasi pakan dan air minum</span>
                                    </li>
                                </ul>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Sesuaikan posisi gambar untuk mendapatkan hasil pembacaan terbaik. Pastikan gambar jelas dan pencahayaan cukup.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
        else:
            # Show instruction when no image is captured
            st.info("👈 Silakan ambil foto atau upload gambar kotoran ayam untuk dianalisis.")
            if scanning_animation:
                st_lottie(scanning_animation, height=200, key="waiting")
        st.markdown('</div>', unsafe_allow_html=True)

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