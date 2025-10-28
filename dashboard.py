import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Model deteksi objek sepak bola
    yolo_model = YOLO("model/Model_Laporan_4_Yolo.pt")
    # Model klasifikasi X-ray
    cnn_model = tf.keras.models.load_model("model/Model_Laporan_2_CNN.h5")
    return yolo_model, cnn_model

yolo_model, cnn_model = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection Dashboard")
st.write("Projek UTS Pemrograman Big Data - Klasifikasi X-ray & Deteksi Objek Sepak Bola")

# Pilih mode
menu = st.sidebar.selectbox("Pilih Mode:", ["Klasifikasi X-ray", "Deteksi Objek Sepak Bola"])

# Upload gambar
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    # ======================
    # Loading Spinner
    # ======================
    with st.spinner("⏳ Data sedang diproses... Mohon tunggu beberapa saat."):
        time.sleep(1)  # simulasi loading

        if menu == "Klasifikasi X-ray":
            # Preprocessing CNN X-ray
            img_resized = img.convert("L").resize((128, 128))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            prediction = cnn_model.predict(img_array)
            prob = prediction[0][0]

            # Interpretasi
            label = "Pneumonia" if prob >= 0.5 else "Normal"

        elif menu == "Deteksi Objek Sepak Bola":
            # Deteksi objek YOLO
            results = yolo_model(np.array(img))

    # ======================
    # Hasil Prediksi / Deteksi
    # ======================
    st.success("✅ Input gambar berhasil diupload!")

    if menu == "Klasifikasi X-ray":
        st.write(f"### Hasil Klasifikasi:")
        st.write(f"Gambar termasuk **{label}** dengan probabilitas **{prob:.2f}**")
        # Narasi tambahan
        if label == "Pneumonia":
            st.info("⚠️ Hasil ini menunjukkan kemungkinan Pneumonia. Silakan konsultasikan ke dokter untuk diagnosis resmi.")
        else:
            st.info("✅ Hasil menunjukkan Normal. Namun, tetap lakukan pemeriksaan rutin jika diperlukan.")

    elif menu == "Deteksi Objek Sepak Bola":
        # Tampilkan gambar dengan bounding box
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

        # Narasi hasil deteksi
        st.write("### Objek Terdeteksi:")
        class_count = {}
        for box in results[0].boxes:
            cls = results[0].names[int(box.cls[0])]
            conf = float(box.conf[0])
            st.write(f"- **{cls}** dengan confidence **{conf:.2f}**")
            if cls in class_count:
                class_count[cls] += 1
            else:
                class_count[cls] = 1

        # Ringkasan narasi singkat
        summary = ", ".join([f"{count} {cls}" for cls, count in class_count.items()])
        st.info(f"Ringkasan: Terdeteksi {summary} dalam gambar.")
