# ------------------------------
# Dashboard UTS 
# ------------------------------
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import os
import io
import json
import time

# ------------------------------
# Config
# ------------------------------
st.set_page_config(page_title="UTS Big Data Dashboard", layout="wide", page_icon="🎓")

# ------------------------------
# Tema Light / Dark (Biru - Ungu Gelap)
# ------------------------------
tema = st.sidebar.radio("Pilih Tema:", ["Terang", "Gelap"])
if tema == "Terang":
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(180deg, #0f172a 0%, #6366f1 100%); color:#FFFFFF;}
    .css-1d391kg {background:#1e293b;} /* sidebar */
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {background:#1E1E2F; color:#FFFFFF;}
    .css-1d391kg {background:#2C2C3A;}
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Header Dashboard
# ------------------------------
st.markdown("""
<div style='padding:15px;border-radius:10px;background: linear-gradient(90deg, #0ea5e9, #6366f1); color:white'>
  <h2>Ujian Tengah Semester - Praktikum Big Data</h2>
  <h3>2208108010017 - Yeni Ckrisdayanti Manalu</h3>
  <p>Dashboard Klasifikasi Gambar & Deteksi Objek</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar / Navigasi dengan ikon
# ------------------------------
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio("Menu", [
    "🏠 Halaman Utama", 
    "🩻 Klasifikasi Gambar", 
    "⚽ Deteksi Objek",
    "🔎 Prediksi", 
    "💬 Feedback",
    "👤 Tentang Penyusun"
])

# ------------------------------
# Paths & Helpers
# ------------------------------
PROJECT_ROOT = os.getcwd()
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

# ------------------------------
# Load Models
# ------------------------------
@st.cache_resource
def load_models():
    cnn_path = os.path.join(MODEL_DIR, "Model_Laporan_2_CNN.h5")
    yolo_path = os.path.join(MODEL_DIR, "Model_Laporan_4_Yolo.pt")
    cnn = tf.keras.models.load_model(cnn_path) if os.path.exists(cnn_path) else None
    yolo = YOLO(yolo_path) if os.path.exists(yolo_path) else None
    return yolo, cnn

yolo_model, cnn_model = load_models()

# ------------------------------
# Halaman Utama
# ------------------------------
if menu == "🏠 Halaman Utama":
    st.title("👋 Selamat datang semuanya!")
    st.markdown("""
Proyek ini dibuat sebagai **Ujian Tengah Semester (UTS) Praktikum Big Data**.  
Dashboard ini mengintegrasikan dua eksperimen computer vision:

- 🩻 **Klasifikasi X-ray (CNN)** — Normal vs Pneumonia  
- ⚽ **Deteksi Objek (YOLO)** — ball, goalkeeper, player, referee  

🔍 **Fitur yang bisa dicoba:**  
- 🏠 Halaman Utama: Beranda dashboard  
- 🩻 Klasifikasi Gambar: Ringkasan model klasifikasi X-ray (untuk menganalisis data kesehatan)  
- ⚽ Deteksi Objek: Ringkasan model deteksi objek sepak bola (mendeteksi pemain dan bola)  
- 🔎 Prediksi: Upload gambar dan lihat prediksi model  
- 💬 Feedback: Berikan saran dan rating  
- 👤 Tentang Penyusun: Info tentang pembuat dashboard
""")

# ------------------------------
# Halaman Klasifikasi Gambar
# ------------------------------
elif menu == "🩻 Klasifikasi Gambar":
    st.title("🩻 Klasifikasi Gambar")
    st.markdown("Di halaman ini, kamu dapat mencoba model klasifikasi gambar X-ray untuk mendeteksi pneumonia.")
    st.markdown("""
**Fungsi Model:**  
Model CNN ini digunakan untuk **menganalisis data kesehatan** dari gambar rontgen X-ray dan membantu mengklasifikasikan pasien sebagai **Normal** atau **Pneumonia**.  

**Dataset:** Chest X-ray Images (Normal & Pneumonia)  
Sumber: [Kaggle](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images)  
- Total gambar: 5.856 (Normal: 1.583, Pneumonia: 4.273)  
- Preprocessing: Rescaling, Augmentasi, Resize 128x128, Grayscale  
- Model: CNN (Conv2D, MaxPooling, Flatten, Dense, Dropout)  
- Evaluasi: Accuracy 93%, Precision & Recall ≥ 60% untuk kelas minor
""")
    st.info("Klik menu 🔎 Prediksi untuk mulai menggunakan model. (Tetap di halaman ini agar dashboard tidak berpindah)")

# ------------------------------
# Halaman Deteksi Objek
# ------------------------------
elif menu == "⚽ Deteksi Objek":
    st.title("⚽ Deteksi Objek")
    st.markdown("""
Halaman ini menggunakan model **YOLOv8n** untuk mendeteksi objek sepak bola.  
Model dapat mengenali **ball, player, goalkeeper, referee** di lapangan.  

**Dataset:** Football Players Detection Dataset  
Sumber: [Kaggle](https://www.kaggle.com/datasets/borhanitrash/football-players-detection-dataset)  
- Jumlah gambar: 312  
- Fungsi: Membantu analisis posisi pemain dan bola di lapangan, evaluasi pertandingan, dll.
""")

# ------------------------------
# Halaman Prediksi
# ------------------------------
elif menu == "🔎 Prediksi":
    st.title("🔎 Prediksi Gambar")
    mode = st.selectbox("Pilih Mode:", ["Klasifikasi X-ray", "Deteksi Objek Sepak Bola"])
    folder_key = "Klasifikasi gambar" if mode.startswith("Klasifikasi") else "Objek deteksi"
    samples = list_samples(folder_key)
    choice = st.selectbox("Atau pilih sample:", ["-- Upload sendiri --"] + samples)
    uploaded = st.file_uploader("Unggah Gambar", type=["jpg","jpeg","png"])
    
    if choice != "-- Upload sendiri --":
        uploaded = io.BytesIO(open(choice, "rb").read())
    
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input Image", use_column_width=True)
        
        if st.button("Jalankan Prediksi"):
            with st.spinner("⏳ Sedang memproses..."):
                time.sleep(1)
                
                if mode == "Klasifikasi X-ray":
                    img_gray = img.convert("L").resize((128,128))
                    arr = keras_image.img_to_array(img_gray)/255.0
                    arr = np.expand_dims(arr,0)
                    pred = cnn_model.predict(arr)
                    prob = float(pred[0][0])
                    label = "Pneumonia" if prob >= 0.5 else "Normal"
                    st.success("✅ Prediksi selesai!")
                    st.markdown(f"**Hasil Prediksi:** {label} — Probabilitas pneumonia: {prob:.2f}")
                
                else:  # Deteksi Objek YOLO
                    results = yolo_model(np.array(img))
                    annotated = results[0].plot()
                    st.success("✅ Prediksi selesai!")
                    st.image(annotated, caption="Hasil Deteksi")
                    
                    st.markdown("### Objek Terdeteksi:")
                    class_count = {}
                    for box in results[0].boxes:
                        cls_idx = int(box.cls[0])
                        cls_name = results[0].names[cls_idx]
                        if cls_name not in class_count:
                            class_count[cls_name] = 0
                        class_count[cls_name] += 1
                    for cls_name, count in class_count.items():
                        st.write(f"- {cls_name}: {count} objek")

# ------------------------------
# Halaman Feedback
# ------------------------------
elif menu == "💬 Feedback":
    st.title("💬 Feedback")
    st.markdown("Silakan berikan saran dan rating untuk dashboard ini.")
    with st.form("form_feedback"):
        nama = st.text_input("Nama")
        rating = st.slider("Rating (1-5)", 1, 5, 3)
        saran = st.text_area("Saran / Komentar")
        submitted = st.form_submit_button("Kirim Feedback")
        if submitted:
            save_feedback({"nama": nama, "rating": rating, "saran": saran, "waktu": time.ctime()})
            st.success("Terima kasih! Feedback berhasil dikirim.")
    
    st.markdown("### Feedback sebelumnya:")
    feedbacks = read_feedback()
    if feedbacks:
        for fb in feedbacks[-5:][::-1]:  # tampilkan 5 terakhir
            st.write(f"- **{fb['nama']}** ({fb['waktu']}): Rating {fb['rating']} — {fb['saran']}")

# ------------------------------
# Halaman Tentang Penyusun
# ------------------------------
elif menu == "👤 Tentang Penyusun":
    st.title("👤 Tentang Penyusun")
    st.markdown("""
Haii, salam kenal! 👋  
Aku **Yeni Ckrisdayanti Manalu**, mahasiswa Statistika USK angkatan 22.  
Projek dashboard ini dibuat untuk memenuhi **UTS Praktikum Big Data**.  

Terima kasih sudah mengunjungi dashboard ini.  
Semoga proyek UTS Big Data ini bermanfaat dan bisa menjadi referensi belajar computer vision dan dashboard interaktif. 😊
""")
