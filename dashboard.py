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
# Pilihan Tema
# ------------------------------
tema = st.sidebar.radio("Pilih Tema:", ["Terang", "Gelap"])
if tema == "Terang":
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%); color:#000;}
    .css-1d391kg {background:#F0F0F0;}
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
  <h2>Ujian Tengah Semester - Praktikum Big Data </h2>
  <h2>2208108010017 - Yeni Ckrisdayanti Manalu </h2>
  <p>Dashboard Klasifikasi Gambar & Deteksi Objek</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar / Navigasi
# ------------------------------
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio("Menu", [
    "Halaman Utama", 
    "Klasifikasi Gambar", 
    "Deteksi Objek",
    "Prediksi", 
    "Feedback",
    "Tentang Penyusun"
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
if menu == "Halaman Utama":
    st.title("👋 Selamat datang semuanya!")
    st.markdown("""
Proyek ini dibuat sebagai **Ujian Tengah Semester (UTS) Praktikum Big Data**.  
Dashboard ini mengintegrasikan dua eksperimen computer vision:

- 🩻 **Klasifikasi X-ray (CNN)** — Normal vs Pneumonia  
- ⚽ **Deteksi Objek (YOLO)** — ball, goalkeeper, player, referee  

🔍 **Fitur yang bisa dicoba:**  
- 🏠 Halaman Utama: Beranda dashboard  
- 🩻 Klasifikasi Gambar: Coba model klasifikasi X-ray  
- ⚽ Deteksi Objek: Coba model deteksi objek sepak bola  
- 🔎 Prediksi: Upload gambar dan lihat prediksi model  
- 💬 Feedback: Berikan saran dan rating  
- 👤 Tentang Penyusun: Info tentang pembuat dashboard
""")

# ------------------------------
# Halaman Klasifikasi Gambar
# ------------------------------
elif menu == "Klasifikasi Gambar":
    st.title("🩻 Klasifikasi Gambar")
    st.markdown("Di halaman ini, kamu dapat mencoba model klasifikasi gambar X-ray untuk mendeteksi pneumonia.")
    st.markdown("""
**Ringkasan Proyek:**  
- Dataset: Chest X-ray Images (Normal dan Pneumonia)  
  Sumber: [Kaggle](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images)  
- Total gambar: 5.856 (Normal: 1.583, Pneumonia: 4.273)  
- Preprocessing: Rescaling, Augmentasi, Resize 128x128, Grayscale  
- Model: CNN (Conv2D, MaxPooling, Flatten, Dense, Dropout)  
- Evaluasi: Accuracy 93%, Precision & Recall ≥ 60% untuk kelas minor
""")
    if st.button("Mulai Menggunakan Model"):
        st.session_state['page'] = 'prediksi'

# ------------------------------
# Halaman Deteksi Objek
# ------------------------------
elif menu == "Deteksi Objek":
    st.title("⚽ Deteksi Objek")
    st.markdown("""
Halaman ini menggunakan model YOLO untuk mendeteksi objek sepak bola.

**Dataset:** Football Players Detection Dataset  
Sumber: [Kaggle](https://www.kaggle.com/datasets/borhanitrash/football-players-detection-dataset)  
Jumlah gambar: 312 (Train: 251, Valid: 43, Test: 18)  
Kelas: ball, goalkeeper, player, referee  
Model: YOLOv8n (non-pretrained)  
Output: Bounding box, class label, confidence score
""")
    st.markdown("""
**Evaluasi Model:**  
- Precision rata-rata: 51,8%  
- Recall rata-rata: 40%  
- mAP50: 41,5%  
- mAP50-95: 21,5%  
Model cukup efisien dan mendeteksi objek dominan (player) dengan baik.
""")

# ------------------------------
# Halaman Prediksi
# ------------------------------
elif menu == "Prediksi":
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
                else:
                    results = yolo_model(np.array(img))
                    annotated = results[0].plot()
                    st.success("✅ Prediksi selesai!")
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
            st.success("Terima kasih! Saranmu sudah tersimpan. 😊")
    feedbacks = read_feedback()
    last3 = feedbacks[-3:][::-1] if feedbacks else []
    if last3:
        st.markdown("### Feedback Terbaru:")
        for fb in last3:
            st.markdown(f"- **{fb['name']}** ({fb['time']}) — Rating: {fb['rating']}/5")
            st.write(fb['suggestion'])

# ------------------------------
# Halaman Tentang Penyusun
# ------------------------------
elif menu == "Tentang Penyusun":
    st.title("👤 Tentang Penyusun")
    st.markdown("""
Haii salam kenal! Aku **Yeni Ckrisdayanti Manalu**, angkatan 22.  
Terima kasih sudah mengunjungi dashboard ini.  
Semoga proyek UTS Big Data ini bermanfaat ya! 😊
""")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
<div style='text-align: center; font-size: 14px;'>
    © 2025 <b>Yeni Ckrisdayanti Manalu</b><br>
    Dengan bimbingan Asisten Bg Diaz & Bg Mus<br>
    🔗 <a href='https://www.linkedin.com/in/yeni-ckrisdayanti-manalu-741a572a9/' target='_blank'>LinkedIn</a> | 
    📸 IG: @yeni.ckrisdayanti_ | Hub: ckyeni
</div>
""", unsafe_allow_html=True)
