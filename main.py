import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Deteksi Penyakit Ayam",
    page_icon="🐔",
    layout="wide"
)


@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan: {MODEL_PATH}")
        return None

    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error("Model gagal dimuat.")
        st.exception(e)
        return None


def load_labels():
    if not os.path.exists(LABELS_PATH):
        return ["Sehat", "NCD", "Koksidiosis", "Salmonela"]

    labels = []
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file.readlines():
            line = line.strip()
            if not line:
                continue

            # Format Teachable Machine biasanya: "0 Sehat"
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line)

    return labels


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = (image_array / 127.5) - 1

    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_image(model, image, labels):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)[0]

    index = int(np.argmax(prediction))
    confidence = float(prediction[index]) * 100

    label = labels[index] if index < len(labels) else f"Kelas {index}"

    return label, confidence, prediction


def show_confidence_gauge(score):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            title={"text": "Confidence Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1E3A8A"},
                "steps": [
                    {"range": [0, 50], "color": "#fee2e2"},
                    {"range": [50, 80], "color": "#fef9c3"},
                    {"range": [80, 100], "color": "#dcfce7"},
                ],
            },
        )
    )

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def get_explanation(label):
    explanations = {
        "Sehat": {
            "title": "✅ Ayam Terindikasi Sehat",
            "text": "Kondisi kotoran terlihat mendekati normal. Tetap perhatikan kebersihan kandang, kualitas pakan, dan air minum."
        },
        "NCD": {
            "title": "⚠️ Terindikasi Newcastle Disease / NCD",
            "text": "NCD adalah penyakit virus yang menular. Gejala dapat berupa gangguan pernapasan, saraf, pencernaan, dan kotoran cair kehijauan. Segera konsultasikan ke dokter hewan."
        },
        "Koksidiosis": {
            "title": "⚠️ Terindikasi Koksidiosis",
            "text": "Koksidiosis menyerang saluran pencernaan. Gejalanya dapat berupa diare, kotoran berdarah, ayam lemas, dan nafsu makan menurun."
        },
        "Salmonela": {
            "title": "⚠️ Terindikasi Salmonela",
            "text": "Salmonela adalah infeksi bakteri yang dapat menyebabkan gangguan pencernaan. Perhatikan kebersihan kandang, pakan, dan air minum."
        },
    }

    return explanations.get(label, {
        "title": f"Hasil: {label}",
        "text": "Sistem berhasil membaca gambar, tetapi informasi penjelasan untuk kelas ini belum tersedia."
    })


def main():
    st.title("🐔 Sistem Deteksi Penyakit Ayam dari Kotoran")
    st.write(
        "Upload atau ambil foto kotoran ayam, lalu sistem akan memprediksi kemungkinan kondisi kesehatan ayam."
    )

    st.warning(
        "Aplikasi ini hanya alat bantu prediksi awal. Hasil akhir tetap perlu dikonfirmasi oleh dokter hewan."
    )

    model = load_ai_model()
    labels = load_labels()

    if model is None:
        st.stop()

    with st.sidebar:
        st.header("Informasi")
        st.write("Model: Teachable Machine / Keras")
        st.write("Ukuran input: 224 x 224")
        st.write("Kelas deteksi:")
        for label in labels:
            st.write(f"- {label}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Gambar")

        input_method = st.radio(
            "Pilih metode input:",
            ["Upload Gambar", "Kamera"],
            horizontal=True
        )

        image_file = None

        if input_method == "Upload Gambar":
            image_file = st.file_uploader(
                "Upload gambar kotoran ayam",
                type=["jpg", "jpeg", "png"]
            )
        else:
            image_file = st.camera_input("Ambil foto kotoran ayam")

        if image_file is not None:
            image = Image.open(image_file)
            st.image(image, caption="Gambar yang dianalisis", use_container_width=True)

    with col2:
        st.subheader("Hasil Deteksi")

        if image_file is None:
            st.info("Silakan upload atau ambil gambar terlebih dahulu.")
            return

        with st.spinner("Menganalisis gambar..."):
            label, confidence, raw_prediction = predict_image(model, image, labels)

        explanation = get_explanation(label)

        if confidence < 60:
            st.warning("Kepercayaan model rendah. Coba gunakan gambar yang lebih jelas, terang, dan fokus.")
        else:
            st.success("Analisis selesai.")

        st.markdown(f"### {explanation['title']}")
        st.write(explanation["text"])

        st.metric("Confidence Score", f"{confidence:.2f}%")
        show_confidence_gauge(confidence)

        with st.expander("Detail Probabilitas"):
            for i, score in enumerate(raw_prediction):
                class_name = labels[i] if i < len(labels) else f"Kelas {i}"
                st.write(f"{class_name}: {score * 100:.2f}%")

    st.markdown("---")
    st.caption("Developed by Galuh Adi Insani")


if __name__ == "__main__":
    main()
