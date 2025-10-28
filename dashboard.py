# dashboard.py
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import cv2
import os
import io
import json
import time

# -----------------------
# CONFIG & STYLE
# -----------------------
st.set_page_config(page_title="UTS - Praktikum Big Data", layout="wide", page_icon="🎓")

# Tema Light / Dark
tema = st.sidebar.radio("Tema:", ["Terang", "Gelap"])
if tema == "Terang":
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%); color:#000;}
    .css-1d391kg {background:#F0F0F0;} /* sidebar */
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {background:#1E1E2F; color:#FFFFFF;}
    .css-1d391kg {background:#2C2C3A;}
    </style>
    """, unsafe_allow_html=True)

# Header Dashboard
st.markdown("""
<div style='padding:15px;border-radius:10px;background: linear-gradient(90deg, #0ea5e9, #6366f1); color:white'>
  <h2>Ujian Tengah Semester - Praktikum Big Data </h2>
  <h3>2208108010017 - Yeni Ckrisdayanti Manalu </h3>
  <p>Dashboard Klasifikasi Gambar & Deteksi Objek</p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Helpers
# -----------------------
PROJECT_ROOT = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "sample_images")
FEEDBACK_FILE = os.path.join(PROJECT_ROOT, "feedback.json")

def ensure_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def save_feedback(entry):
    ensure_feedback_file()
    with open(FEEDBACK_FILE, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_feedback():
    ensure_feedback_file()
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def list_samples(subfolder):
    folder = os.path.join(SAMPLES_DIR, subfolder)
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    return [os.path.join(folder, f) for f in files]

def pil_to_bytes(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

def cv2_imencode_png(cv2_img):
    success, encoded = cv2.imencode(".png", cv2_img)
    if not success:
        return None
    return encoded.tobytes()

# -----------------------
# Load Models (cached)
# -----------------------
@st.cache_resource
def load_models():
    cnn_path = os.path.join(MODEL_DIR, "Model_Laporan_2_CNN.h5")
    yolo_path = os.path.join(MODEL_DIR, "Model_Laporan_4_Yolo.pt")
    cnn = None
    yolo = None
    if os.path.exists(cnn_path):
        try:
            cnn = tf.keras.models.load_model(cnn_path)
        except Exception as e:
            st.error(f"Gagal load CNN model: {e}")
    else:
        st.warning("Model CNN (.h5) tidak ditemukan di folder model/.")
    if os.path.exists(yolo_path):
        try:
            yolo = YOLO(yolo_path)
        except Exception as e:
            st.error(f"Gagal load YOLO model: {e}")
    else:
        st.warning("Model YOLO (.pt) tidak ditemukan di folder model/.")
    return yolo, cnn

yolo_model, cnn_model = load_models()

# ------------------------------
# Menu Sidebar / Navigasi
# ------------------------------
st.sidebar.title("Navigasi Fitur")
menu = st.sidebar.radio("Menu", [
    "Halaman Utama", 
    "Klasifikasi Gambar", 
    "Deteksi Objek",
    "Prediksi", 
    "Feedback",
    "Tentang Penyusun"
])

# ------------------------------
# Session state untuk navigasi tombol
# ------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = None

# ------------------------------
# Halaman Utama
# ------------------------------
if menu == "Halaman Utama":
    st.title("Selamat Datang di Dashboard UTS 👋")
    st.markdown("""
    Proyek ini dibuat sebagai **Ujian Tengah Semester (UTS)** Praktikum Big Data.  
    Dashboard ini menyatukan dua eksperimen computer vision:

    🩻 **Klasifikasi X-ray (CNN)**: Normal vs Pneumonia  
    ⚽ **Deteksi Objek (YOLO)**: ball, player, goalkeeper, referee

    🔍 Pilih fitur dari menu navigasi di samping kiri untuk dicoba.
    """)

# ------------------------------
# Halaman Klasifikasi Gambar
# ------------------------------
elif menu == "Klasifikasi Gambar":
    st.title("🩻 Klasifikasi Gambar X-ray")
    st.markdown("""
    Di halaman ini, kamu dapat mencoba model klasifikasi gambar X-ray untuk mendeteksi pneumonia.

    **Ringkasan Proyek:**
    - **Dataset**: [Chest X-ray Images](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images)
    - **Jumlah gambar**: 5.856 (Normal: 1.583, Pneumonia: 4.273)
    - **Preprocessing**: Rescaling, Augmentasi, Resize 128x128, Grayscale
    - **Model**: CNN dengan beberapa Conv2D, MaxPooling, Flatten, Dense, Dropout
    - **Evaluasi**: Accuracy 93%, Precision & Recall ≥ 60% untuk kelas minor
    """)
    
    if st.button("Mulai Menggunakan Model"):
        st.session_state['page'] = 'prediksi'
        st.experimental_rerun()

# ------------------------------
# Halaman Deteksi Objek
# ------------------------------
elif menu == "Deteksi Objek":
    st.title("⚽ Deteksi Objek Sepak Bola")
    st.markdown("""
    Dataset: [Football Players Detection Dataset](https://www.kaggle.com/datasets/borhanitrash/football-players-detection-dataset)  
    Dataset ini dibuat untuk mendeteksi objek utama dalam pertandingan sepak bola seperti **ball, player, goalkeeper, referee**.

    - Jumlah gambar: 312 (Train: 251, Valid: 43, Test: 18)
    - Model: YOLOv8n dari awal (non-pretrained)
    - Output: Bounding box, class label, confidence score

    **Evaluasi Model:**
    - Precision rata-rata: 51,8%
    - Recall rata-rata: 40%
    - mAP50: 41,5%
    - mAP50-95: 21,5%
    Model cukup efisien dan mendeteksi objek dominan (player) dengan baik.
    """)

    if st.button("Mulai Menggunakan Model"):
        st.session_state['page'] = 'prediksi'
        st.experimental_rerun()

# ------------------------------
# Halaman Prediksi
# ------------------------------
elif menu == "Prediksi":
    st.title("🔍 Prediksi")
    mode = st.selectbox("Pilih Mode:", ["Klasifikasi X-ray", "Deteksi Objek Sepak Bola"])
    
    uploaded = st.file_uploader("Unggah Gambar", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input Image", use_column_width=True)
        
        if st.button("Jalankan Prediksi"):
            with st.spinner("⏳ Data sedang diproses..."):
                time.sleep(1)
                if mode == "Klasifikasi X-ray":
                    img_gray = img.convert("L").resize((128,128))
                    arr = keras_image.img_to_array(img_gray)/255.0
                    arr = np.expand_dims(arr,0)
                    pred = cnn_model.predict(arr)
                    prob = float(pred[0][0])
                    label = "Pneumonia" if prob >= 0.5 else "Normal"
                    st.success("✅ Gambar berhasil dianalisis!")
                    st.markdown(f"**Hasil Prediksi:** {label} — Probabilitas pneumonia: {prob:.2f}")
                else:
                    results = yolo_model(np.array(img))
                    annotated = results[0].plot()
                    st.success("✅ Gambar berhasil dianalisis!")
                    st.image(annotated, caption="Hasil Deteksi")
                    
                    st.markdown("### Objek Terdeteksi:")
                    class_count = {}
                    for box in results[0].boxes:
                        cls_idx = int(box.cls[0])
                        cls_name = results[0].names[cls_idx]
                        conf = float(box.conf[0])
                        st.write(f"- {cls_name} (confidence {conf:.2f})")
                        class_count[cls_name] = class_count.get(cls_name,0)+1
                    st.info(f"Ringkasan: {', '.join([f'{v} {k}' for k,v in class_count.items()]) if class_count else 'Tidak ada objek terdeteksi.'}")

# ------------------------------
# Halaman Feedback
# ------------------------------
elif menu == "Feedback":
    st.title("💬 Feedback & Saran")
    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Nama (opsional)")
        rating = st.slider("Seberapa puas?", 1, 5, 4)
        suggestion = st.text_area("Saran / komentar")
        submitted = st.form_submit_button("Kirim")
        
        if submitted:
            save_feedback({
                "name": name if name else "Anonim",
                "rating": rating,
                "suggestion": suggestion,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Terima kasih! Saran tersimpan.")

# ------------------------------
# Halaman Tentang Penyusun
# ------------------------------
elif menu == "Tentang Penyusun":
    st.title("👤 Tentang Penyusun")
    st.markdown("""
    Haii salam kenal! Aku **Yeni Ckrisdayanti Manalu**, angkatan 22.  
    Terima kasih sudah mengunjungi dashboard ku.  
    Ini adalah proyek UTS Big Data, semoga bermanfaat ya 😊
    """)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 14px;'>
        © 2025 <b>Yeni Ckrisdayanti Manalu</b><br>
        Dengan bimbingan Asisten Bg Diaz & Bg Mus<br>
        <a href='https://www.linkedin.com/in/yeni-ckrisdayanti-manalu-741a572a9/' target='_blank'>
        🔗 LinkedIn</a><br>
        Instagram: @yeni.ckrisdayanti_<br>
        Hubungi: ckyeni
    </div>
    """, unsafe_allow_html=True)
