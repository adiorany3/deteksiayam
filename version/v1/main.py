# Developed by Galuh Adi Insani
# Dimohon jangan hilangkan pada bagian ini untuk menghargai hasil kerja keras developer

# Import necessary libraries
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

# Define current_year at the module level
current_year = datetime.datetime.now().year

# Page configuration
st.set_page_config(
    page_title="Deteksi Penyakit Ayam",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        padding: 0.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, rgba(219,234,254,0.3) 0%, rgba(191,219,254,0.3) 100%);
        border-radius: 10px;
    }
    
    /* Card styling */
    .disease-card {
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    .disease-card:hover {
        transform: translateY(-2px);
    }
    
    /* Card variants - without background colors, more subtle styling */
    .healthy-card {
        background: white;
        border-left: 4px solid #00000F;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .ncd-card {
        background: white;
        border-left: 4px solid #00000F;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .coccidiosis-card {
        background: white;
        border-left: 4px solid #00000F;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .salmonella-card {
        background: linear-gradient(135deg, #dbeafe 0%, #00000F 100%);
        border-left: 5px solid #3b82f6;
    }
    
    /* Text elements */
    .disease-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #111827;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1E3A8A; 
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1E40AF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Make sidebar more elegant */
    .css-1d391kg {
        background-color: #f1f5f9;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid #e5e7eb;
        color: #6b7280;
    }
    
    /* Remove default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Improve expandable sections */
    .streamlit-expanderHeader {
        background-color: #f8fafc !important;
        font-weight: 600 !important;
        color: #1E3A8A !important;
    }
    
    /* Add animation to cards */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .disease-card {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* Improve confidence score display */
    .confidence-display {
        text-align: center;
        font-size: 1.25rem;
        font-weight: 600;
        color: #1E3A8A;
        margin: 0.5rem 0;
    }
    .confidence-bar {
        height: 6px;
        background-color: #e5e7eb;
        border-radius: 3px;
        margin: 0.5rem 0;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 3px;
    }
    
    /* Camera input styling */
    .camera-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar contents with improved styling
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🤖 Deteksi Penyakit Ayam</h2>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ Tentang Aplikasi", expanded=True):
        st.markdown('''
        ### Tentang Software
        
        Aplikasi ini dirancang untuk memprediksi penyakit pada unggas berdasarkan visualisasi kotoran ayam menggunakan teknologi computer vision dan machine learning.
        
        **Penyakit yang dapat dideteksi:**
        - ✅ **Sehat** - Kondisi normal
        - 🦠 **NCD** - Newcastle Disease 
        - 🔬 **Koksidiosis** - Infeksi parasit
        - 🧫 **Salmonela** - Infeksi bakteri
        ''')
    
    with st.expander("📋 Cara Penggunaan", expanded=False):
        st.markdown('''
        1. 📸 Foto kotoran ayam dengan kamera
        2. 🔍 Pastikan pencahayaan memadai
        3. 📊 Sistem akan menganalisis dan menampilkan hasil
        
        > ⚠️ Software memberikan prediksi awal, hasil akhir harus dikonfirmasi oleh dokter hewan.
        ''')
        
    with st.expander("🔬 Sumber Data", expanded=False):
        st.markdown('''
        Data dikembangkan dengan memanfaatkan database Kaggle yang diproses menggunakan teknologi machine learning.
        
        📊 [Chicken Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/allandclive/chicken-disease-1)
        ''')
    
    st.markdown("---")
    st.markdown("<div style='text-align: center;'><h4>Developer Contact</h4></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><a href='https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/' target='_blank'>🔗 Galuh Adi Insani</a></div>", unsafe_allow_html=True)

# Create custom function for displaying confidence score
def display_confidence(score):
    color = "#22c55e" if score > 90 else "#eab308" if score > 70 else "#ef4444"
    
    html = f"""
    <div class="confidence-display">
        Confidence Score: {score:.2f}%
    </div>
    <div class="confidence-bar">
        <div class="confidence-bar-fill" style="width: {score}%; background-color: {color};"></div>
    </div>
    """
    
    return st.markdown(html, unsafe_allow_html=True)

# Plotly gauge chart for confidence score
def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 16, 'color': "#1E3A8A"}},
        number = {'font': {'size': 20, 'color': "#1E3A8A"}, 'suffix': "%", 'valueformat': '.2f'},  # Format with 2 decimals
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1E3A8A"},
            'bar': {'color': "#1E3A8A" if score > 90 else "#eab308" if score > 70 else "#ef4444"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 80], 'color': '#fef9c3'},
                {'range': [80, 100], 'color': '#dcfce7'}
            ],
            'threshold': {
                'line': {'color': "#16a34a", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

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
    # Main header
    st.markdown("<h1 class='main-header'>🔍 Sistem Deteksi Penyakit Ayam</h1>", unsafe_allow_html=True)
    
    # Create two columns for the main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3>📸 Ambil Gambar</h3>", unsafe_allow_html=True)
        st.markdown("<div class='camera-container'>", unsafe_allow_html=True)
        # Camera input with improved styling
        image = st.camera_input(label="Capture Image", key="First Camera", label_visibility="collapsed")
        
        # Add option to upload image
        with st.expander("📤 Upload Gambar"):
            uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None and image is None:
                image = uploaded_file
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Load model
    model_eval = load_models()
    
    # Results section
    with col2:
        st.markdown("<h3>🔬 Hasil Analisis</h3>", unsafe_allow_html=True)
        
        if model_eval is None:
            st.error("Model tidak dapat dimuat. Silakan periksa file model Anda.")
            return
        
        if image:
            # Display uploaded/captured image
            with st.expander("🖼️ Gambar Input", expanded=True):
                st.image(image, caption="Gambar Kotoran Ayam", use_container_width=True)
            
            with st.spinner("Menganalisis gambar..."):
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
                confidence_percent = confidence_score * 100  # Keep as float, don't convert to int
                
                # Display results based on prediction
                if confidence_score > 0.7:  # Slightly lower threshold for better usability
                    if index == 0:
                        st.markdown(f"""
                        <div class="disease-card healthy-card">
                            <div class="disease-title" style="color: #000000;">✅ Sehat</div>
                            <p style="color: #000000;">Kotoran ayam yang sehat memiliki ciri-ciri tertentu yang mencerminkan kesehatan pencernaan dan kondisi ayam secara keseluruhan.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif index == 1:
                        st.markdown(f"""
                        <div class="disease-card ncd-card">
                            <div class="disease-title" style="color: #000000;">🦠 Newcastle Disease (NCD)</div>
                            <p style="color: #000000;">Penyakit virus yang sangat menular. Gejala: Gangguan pernapasan, saraf, dan pencernaan. Kotoran bisa berwarna hijau atau kuning, berair, dan mengandung darah. Pencegahan: Vaksinasi rutin.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif index == 2:
                        st.markdown(f"""
                        <div class="disease-card coccidiosis-card">
                            <div class="disease-title" style="color: #000000;">🔬 Koksidiosis</div>
                            <p style="color: #000000;">Penyakit protozoa yang menyerang usus. Gejala: Kotoran berdarah, diare, penurunan nafsu makan, dan kelemahan. Pencegahan: Menjaga kebersihan kandang, pemberian antikoksidia.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif index == 3:
                        st.markdown(f"""
                        <div class="disease-card salmonella-card">
                            <div class="disease-title" style="color: #000000;">🧫 Salmonela</div>
                            <p style="color: #000000;">Infeksi bakteri yang dapat menyebabkan gangguan pencernaan. Gejala: Diare, kotoran berwarna hijau atau kuning, penurunan nafsu makan, dan demam. Pencegahan: Menjaga kebersihan kandang dan pakan, pemberian antibiotik jika diperlukan.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display confidence score
                    display_confidence(confidence_percent)
                    
                    # Display gauge chart
                    st.plotly_chart(create_gauge_chart(confidence_percent), use_container_width=True)
                else:
                    st.warning("Sesuaikan posisi gambar, untuk mendapatkan hasil pembacaan terbaik")

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