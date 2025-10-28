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
tema = st.sidebar.radio("Pilih Tema:", ["Terang", "Gelap"])
if tema == "Terang":
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%); color:#000;}
    .css-1d391kg {background:#F0F0F0;} /* sidebar */
    h1, h2, h3, h4, h5, h6 {color:#0a0a0a;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {background:#1E1E2F; color:#FFFFFF;}
    .css-1d391kg {background:#2C2C3A;}
    h1, h2, h3, h4, h5, h6 {color:#FFFFFF;}
    </style>
    """, unsafe_allow_html=True)

# -----------------------
# HEADER / KOP
# -----------------------
st.markdown("""
<div style='padding:20px;border-radius:12px;background: linear-gradient(90deg, #0ea5e9, #6366f1); color:white; text-align:center'>
  <h1>🎓 Ujian Tengah Semester - Praktikum Big Data</h1>
  <h2>2208108010017 - Yeni Ckrisdayanti Manalu</h2>
  <p>Dashboard Interaktif: Klasifikasi Gambar & Deteksi Objek</p>
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

# -----------------------
# Load Models
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

# -----------------------
# Sidebar / Navigation
# -----------------------
st.sidebar.markdown("### 📂 Menu")
page = st.sidebar.radio("", ("Home", "Eksplorasi Dataset", "Prediksi", "Feedback", "Tentang"))

# -----------------------
# Session State
# -----------------------
if "pred_mode" not in st.session_state:
    st.session_state["pred_mode"] = "Klasifikasi"
if "go_to_prediksi" not in st.session_state:
    st.session_state["go_to_prediksi"] = False

# -----------------------
# PAGE: HOME
# -----------------------
if page == "Home":
    st.markdown("## 👋 Halo! Selamat datang di Dashboard UTS")
    st.markdown("""
    Proyek ini dibuat sebagai **Ujian Tengah Semester Praktikum Big Data**.  
    Dashboard menyatukan dua eksperimen computer vision:
    - 🩻 **Klasifikasi X-ray (CNN)**: Normal vs Pneumonia
    - ⚽ **Deteksi Objek (YOLO)**: ball, player, goalkeeper, referee
    """)
    
    st.markdown("### 🔍 Pilih fitur untuk dicoba:")
    c1, c2 = st.columns(2)
    
    if c1.button("🩻 Klasifikasi Gambar"):
        st.session_state["pred_mode"] = "Klasifikasi"
        st.session_state["go_to_prediksi"] = True
    if c2.button("⚽ Deteksi Objek"):
        st.session_state["pred_mode"] = "Deteksi"
        st.session_state["go_to_prediksi"] = True

    if st.session_state["go_to_prediksi"]:
        st.session_state["go_to_prediksi"] = False
        st.experimental_rerun()

# -----------------------
# PAGE: EKSPLORASI DATASET
# -----------------------
elif page == "Eksplorasi Dataset":
    st.markdown("## 📊 Eksplorasi Dataset")
    
    # Klasifikasi X-ray
    st.markdown("### 🩻 Klasifikasi X-ray")
    x_samples = list_samples("Klasifikasi gambar")
    if x_samples:
        st.markdown(f"Jumlah sample: **{len(x_samples)}**")
        st.image(x_samples[:6], width=140)
    else:
        st.info("Tidak ada sample X-ray.")
    
    # Deteksi Objek
    st.markdown("### ⚽ Deteksi Objek Sepak Bola")
    y_samples = list_samples("Objek deteksi")
    if y_samples:
        st.markdown(f"Jumlah sample: **{len(y_samples)}**")
        st.image(y_samples[:6], width=140)
    else:
        st.info("Tidak ada sample objek deteksi.")

# -----------------------
# PAGE: PREDIKSI
# -----------------------
elif page == "Prediksi":
    st.markdown("## 🔎 Prediksi (Upload / Pilih Sample)")
    mode = st.selectbox("Pilih Mode:", ["Klasifikasi X-ray", "Deteksi Objek Sepak Bola"],
                        index=0 if st.session_state.get("pred_mode", "") != "Deteksi" else 1)
    
    folder_key = "Klasifikasi gambar" if mode.startswith("Klasifikasi") else "Objek deteksi"
    samples = list_samples(folder_key)
    choice = st.selectbox("Atau pilih sample:", ["-- Upload sendiri --"] + samples)
    
    uploaded = st.file_uploader("Unggah Gambar", type=["jpg","jpeg","png"])
    if choice != "-- Upload sendiri --":
        uploaded = io.BytesIO(open(choice, "rb").read())
    
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="📷 Input Image", use_column_width=True)
        
        if st.button("⚡ Jalankan Prediksi"):
            with st.spinner("⏳ Sedang memproses..."):
                prog = st.progress(0)
                for i in range(4):
                    time.sleep(0.25)
                    prog.progress((i+1)*25)
                
                if mode == "Klasifikasi X-ray":
                    img_gray = img.convert("L").resize((128,128))
                    arr = keras_image.img_to_array(img_gray)/255.0
                    arr = np.expand_dims(arr,0)
                    pred = cnn_model.predict(arr)
                    prob = float(pred[0][0])
                    label = "🦠 Pneumonia" if prob >= 0.5 else "✅ Normal"
                    st.success(f"**Hasil Prediksi:** {label} — Probabilitas pneumonia: {prob:.2f}")
                else:
                    results = yolo_model(np.array(img))
                    annotated = results[0].plot()
                    st.success("✅ Hasil Deteksi Objek")
                    st.image(annotated, caption="Hasil Deteksi YOLO")
                    
                    st.markdown("### Objek Terdeteksi:")
                    class_count = {}
                    for box in results[0].boxes:
                        cls_idx = int(box.cls[0])
                        cls_name = results[0].names[cls_idx]
                        conf = float(box.conf[0])
                        st.write(f"- {cls_name} (confidence {conf:.2f})")
                        class_count[cls_name] = class_count.get(cls_name,0)+1
                    st.info(f"Ringkasan: {', '.join([f'{v} {k}' for k,v in class_count.items()]) if class_count else 'Tidak ada objek terdeteksi.'}")

# -----------------------
# PAGE: FEEDBACK
# -----------------------
elif page == "Feedback":
    st.markdown("## 📝 Feedback & Saran")
    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Nama (opsional)")
        rating = st.slider("Seberapa puas?", 1, 5, 4)
        suggestion = st.text_area("Saran / komentar")
        submitted = st.form_submit_button("Kirim ✉️")
        
        if submitted:
            save_feedback({
                "name": name if name else "Anonim",
                "rating": rating,
                "suggestion": suggestion,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Terima kasih! 🙌 Saran tersimpan.")

    feedbacks = read_feedback()
    if feedbacks:
        st.markdown("### 🔹 Review Terbaru")
        for fb in feedbacks[-3:][::-1]:
            st.info(f"**{fb['name']}** ({fb['time']}) — Rating: {fb['rating']}/5\n💬 {fb['suggestion']}")

# -----------------------
# PAGE: TENTANG
# -----------------------
elif page == "Tentang":
    st.markdown("## ℹ️ Tentang & Kontak")
    st.markdown("""
    **Proyek:** UTS Praktikum Big Data  
    **Pembuat:** Yeni Ckrisdayanti Manalu (Statistika 2022) 🎓  
    **Model:** CNN (.h5) & YOLOv8n (.pt)  
    **Dataset:** Chest X-ray, Football Players Detection  
    **Bimbingan:** Asistensi praktikum oleh Bg Diaz & Bg Mus
    """)
    st.markdown("---")
    st.markdown("**Link & Kontak** 🌐")
    st.markdown("""
    - GitHub: (repo)  
    - LinkedIn: [Yeni Ckrisdayanti](https://www.linkedin.com/in/yeni-ckrisdayanti-manalu-741a572a9/)  
    - Instagram: (opsional)
    """)
