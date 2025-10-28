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
import random
import matplotlib.pyplot as plt

# -----------------------
# CONFIG & STYLE
# -----------------------
st.set_page_config(page_title="UTS - Praktikum Big Data", layout="wide", page_icon="🎓")
CSS = """
<style>
/* background & container */
.stApp {
    background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);
}

/* header style */
.header {
    padding: 18px 24px;
    border-radius: 8px;
    background: linear-gradient(90deg, rgba(14,165,233,0.07), rgba(99,102,241,0.04));
    border-left: 6px solid rgba(99,102,241,0.8);
}
.kop-title {
    font-size:26px;
    font-weight:700;
    margin-bottom:4px;
}
.kop-sub {
    color: #334155;
    margin-top: -6px;
    font-size:14px;
}

/* card style */
.card {
    padding: 12px;
    border-radius: 10px;
    background: #fff;
    box-shadow: 0 1px 8px rgba(15,23,42,0.06);
    border: 1px solid rgba(15,23,42,0.04);
}
.small-muted { color:#556; font-size:13px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

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
    # include common image extensions
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
    # CNN model
    cnn_path = os.path.join(MODEL_DIR, "Model_Laporan_2_CNN.h5")
    yolo_path = os.path.join(MODEL_DIR, "Model_Laporan_4_Yolo.pt")
    cnn = None
    yolo = None
    # Load CNN safely
    if os.path.exists(cnn_path):
        try:
            cnn = tf.keras.models.load_model(cnn_path)
        except Exception as e:
            st.error(f"Gagal load CNN model: {e}")
    else:
        st.warning("Model CNN (.h5) tidak ditemukan di folder model/. Pastikan file ada.")

    # Load YOLO
    if os.path.exists(yolo_path):
        try:
            yolo = YOLO(yolo_path)
        except Exception as e:
            st.error(f"Gagal load YOLO model: {e}")
    else:
        st.warning("Model YOLO (.pt) tidak ditemukan di folder model/. Pastikan file ada.")
    return yolo, cnn

yolo_model, cnn_model = load_models()

# -----------------------
# Sidebar / Navigation
# -----------------------
st.sidebar.write("")  # kosong tapi tetap aman
st.sidebar.markdown("### 📂 Menu")
page = st.sidebar.radio("", ("Home", "Eksplorasi Dataset", "Prediksi", "Feedback", "Tentang"))

# Optional quick links
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Actions**")
if st.sidebar.button("Buka Klasifikasi X-ray"):
    page = "Prediksi"
    st.experimental_rerun()  # rerun to go to page (if you prefer)
st.sidebar.markdown("")

# -----------------------
# Header Kop (global)
# -----------------------
st.markdown(
    f"""
    <div class="header">
      <div class="kop-title">Ujian Tengah Semester - Praktikum Big Data</div>
      <div class="kop-sub">Dashboard Klasifikasi Citra & Deteksi Objek Berbasis Deep Learning</div>
    </div>
    """, unsafe_allow_html=True
)

# -----------------------
# PAGE: HOME
# -----------------------
if page == "Home":
    st.write("")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("## Selamat datang 👋")
        st.markdown(
            """
            Proyek ini dibuat sebagai tugas **Ujian Tengah Semester (UTS)** Praktikum Big Data.
            Dashboard ini mengintegrasikan **dua eksperimen computer vision**:
            1. **Klasifikasi X-ray (CNN)** — mendeteksi `Normal` vs `Pneumonia` dari citra X-ray dada.  
            2. **Deteksi Objek (YOLO)** — mendeteksi objek terkait sepak bola (ball, goalkeeper, player, referee).
            """
        )
        st.markdown("### Penasaran? Pilih salah satu untuk mencoba:")
        c1, c2 = st.columns(2)
        if c1.button("🩻 Klasifikasi Gambar"):
            page = "Prediksi"
            st.session_state["pred_mode"] = "Klasifikasi"
            st.experimental_rerun()
        if c2.button("⚽ Deteksi Objek"):
            page = "Prediksi"
            st.session_state["pred_mode"] = "Deteksi"
            st.experimental_rerun()

        st.markdown("---")
        st.markdown("### Preview Halaman")
        st.markdown(
            "- **Eksplorasi Dataset** — ringkasan distribusi dan contoh gambar.\n"
            "- **Prediksi** — upload / pilih sample → jalankan model → lihat hasil + narasi.\n"
            "- **Feedback** — isi saran, lihat 3 saran terakhir.\n"
            "- **Tentang** — profil pembuat & link."
        )
    with col2:
        st.image(os.path.join(PROJECT_ROOT, "sample_images", "Klasifikasi gambar", os.listdir(os.path.join(SAMPLES_DIR, "Klasifikasi gambar"))[0]) if os.path.isdir(os.path.join(SAMPLES_DIR, "Klasifikasi gambar")) and len(os.listdir(os.path.join(SAMPLES_DIR, "Klasifikasi gambar"))) else None,
                 caption="Contoh dataset", use_column_width=True)

    st.markdown("---")
    st.markdown("### Sekilas Data (ringkas)")
    # show small chart for X-ray distribution if images present
    x_samples = list_samples("Klasifikasi gambar")
    if x_samples:
        # try to infer counts from filenames by folder content
        # for user-provided dataset counts we show simple counts
        # Here we only display sample counts (since full dataset might be elsewhere)
        counts = {"Normal": 0, "Pneumonia": 0}
        for p in x_samples:
            # naive heuristic: filename contains word normal/pneumonia
            lower = os.path.basename(p).lower()
            if "normal" in lower:
                counts["Normal"] += 1
            elif "pneumonia" in lower:
                counts["Pneumonia"] += 1
        # fallback if zero
        if counts["Normal"] + counts["Pneumonia"] == 0:
            counts["Normal"] = len(x_samples)//2
            counts["Pneumonia"] = len(x_samples) - counts["Normal"]

        fig, ax = plt.subplots(figsize=(4,3))
        ax.pie([counts["Normal"], counts["Pneumonia"]], labels=["Normal","Pneumonia"], autopct="%1.0f%%")
        ax.set_title("Distribusi kelas (sample)")
        st.pyplot(fig)
    else:
        st.info("Belum ada sample gambar klasifikasi pada folder `sample_images/Klasifikasi gambar/`.")

# -----------------------
# PAGE: EKSPLORASI DATASET
# -----------------------
elif page == "Eksplorasi Dataset":
    st.markdown("## Eksplorasi Dataset")
    st.markdown("Di halaman ini ditampilkan ringkasan dataset yang digunakan untuk masing-masing model.")
    # X-ray
    st.markdown("### Klasifikasi X-ray (Chest X-ray Images)")
    st.markdown("- Sumber: Kaggle (Chest X-ray Images).")
    st.markdown("- Kelas: Normal, Pneumonia.")
    x_samples = list_samples("Klasifikasi gambar")
    if x_samples:
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown(f"Jumlah sample di folder: **{len(x_samples)}**")
            if len(x_samples) > 0:
                st.image(x_samples[:6], caption=[os.path.basename(p) for p in x_samples[:6]], width=120)
        with col2:
            # show bar chart simple
            counts = {"Normal": 0, "Pneumonia": 0}
            for p in x_samples:
                lower = os.path.basename(p).lower()
                if "normal" in lower:
                    counts["Normal"] += 1
                elif "pneumonia" in lower:
                    counts["Pneumonia"] += 1
            if counts["Normal"] + counts["Pneumonia"] == 0:
                counts["Normal"] = len(x_samples)//2
                counts["Pneumonia"] = len(x_samples) - counts["Normal"]
            st.bar_chart({"Kelas": list(counts.values())}, use_container_width=True)
    else:
        st.info("Tidak ada sample X-ray di folder sample_images/Klasifikasi gambar/")

    st.markdown("---")
    # YOLO dataset
    st.markdown("### Deteksi Objek Sepak Bola (Football Players Detection Dataset)")
    st.markdown("- Sumber: Kaggle / Roboflow.")
    y_samples = list_samples("Objek deteksi")
    if y_samples:
        st.markdown(f"Jumlah sample di folder: **{len(y_samples)}**")
        st.image(y_samples[:6], caption=[os.path.basename(p) for p in y_samples[:6]], width=140)
    else:
        st.info("Tidak ada sample gambar deteksi di folder sample_images/Objek deteksi/")

# -----------------------
# PAGE: PREDIKSI
# -----------------------
elif page == "Prediksi":
    st.markdown("## Prediksi (Upload / Pilih Sample)")
    st.markdown("Pilih mode: Klasifikasi X-ray atau Deteksi Objek. Kamu juga bisa memilih sample dari folder contoh.")
    # mode selector (allow pre-selection from Home buttons)
    pre_mode = st.session_state.get("pred_mode", None)
    mode = st.selectbox("Pilih Mode Prediksi:", ["Klasifikasi X-ray", "Deteksi Objek Sepak Bola"], index=0 if pre_mode!="Deteksi" else 1)

    sample_folder_key = "Klasifikasi gambar" if mode.startswith("Klasifikasi") else "Objek deteksi"
    samples = list_samples(sample_folder_key)
    sample_paths = ["-- Upload sendiri --"] + samples
    choice = st.selectbox("Atau pilih salah satu sample (opsional):", sample_paths)

    uploaded = st.file_uploader("Unggah Gambar (jpg/png)", type=["jpg","jpeg","png"])
    # if user chose sample, override uploaded
    if choice != "-- Upload sendiri --":
        try:
            uploaded = open(choice, "rb")
            # wrap as BytesIO so PIL can read
            uploaded = io.BytesIO(uploaded.read())
        except Exception:
            uploaded = None

    if uploaded is None:
        st.info("Pilih sample atau unggah gambar untuk memulai prediksi.")
    else:
        img = Image.open(uploaded).convert("RGB")
        colA, colB = st.columns([1,1])
        with colA:
            st.image(img, caption="Input Image", use_column_width=True)
        # Run prediction with spinner + progress
        run_btn = st.button("🔎 Jalankan Prediksi")
        if run_btn:
            # spinner + progress simulation while running model
            with st.spinner("⏳ Data sedang diproses... Mohon tunggu."):
                prog = st.progress(0)
                for i in range(4):
                    time.sleep(0.25)
                    prog.progress((i+1)*20)
                # run model depending on mode
                if mode == "Klasifikasi X-ray":
                    # preprocess for CNN: grayscale 128x128
                    img_gray = img.convert("L").resize((128,128))
                    arr = keras_image.img_to_array(img_gray)
                    arr = np.expand_dims(arr, 0) / 255.0
                    try:
                        pred = cnn_model.predict(arr)
                        prob = float(pred[0][0])
                        label = "Pneumonia" if prob >= 0.5 else "Normal"
                    except Exception as e:
                        st.error(f"Gagal melakukan prediksi CNN: {e}")
                        label = None
                        prob = None
                else:
                    # YOLO inference
                    try:
                        results = yolo_model(np.array(img))
                    except Exception as e:
                        st.error(f"Gagal melakukan inferensi YOLO: {e}")
                        results = None

            # After spinner: show results & narrative
            st.success("✅ Input gambar berhasil dianalisis!")
            if mode == "Klasifikasi X-ray" and label is not None:
                col1, col2 = st.columns([1,1])
                with col1:
                    st.image(img_gray, caption="Preprocessed (grayscale 128x128)", use_column_width=True)
                with col2:
                    st.markdown("### Hasil Klasifikasi")
                    st.markdown(f"**{label}**  — Probabilitas pneumonia: **{prob:.2f}**")
                    if label == "Pneumonia":
                        st.warning("⚠️ Model mendeteksi kemungkinan Pneumonia. Hasil ini hanya sebagai referensi; konsultasikan ke tenaga medis untuk diagnosis akhir.")
                    else:
                        st.info("✅ Hasil menunjukkan Normal. Jika ada keluhan, tetap disarankan pemeriksaan lanjutan.")
                # small details
                st.markdown("---")
                st.markdown("**Ringkasan singkat:**")
                st.write(f"Model CNN: struktur custom, input 128×128 grayscale, trained dengan augmentasi dan early stopping. Akurasi laporan: ~93% (contoh).")
            elif mode == "Deteksi Objek Sepak Bola":
                if results is not None:
                    # display annotated image
                    annotated = results[0].plot()  # returns numpy array (BGR or RGB)
                    # ultralytics returns RGB numpy; convert to BGR for cv2 encode if needed
                    if isinstance(annotated, np.ndarray):
                        annotated_cv = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    else:
                        # fallback convert from PIL
                        annotated_cv = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
                    col1, col2 = st.columns([1,1])
                    with col1:
                        st.image(annotated, caption="Hasil Deteksi (annotated)", use_column_width=True)
                    with col2:
                        st.markdown("### Objek Terdeteksi")
                        class_count = {}
                        try:
                            for box in results[0].boxes:
                                cls_idx = int(box.cls[0])
                                cls_name = results[0].names[cls_idx]
                                conf = float(box.conf[0])
                                st.write(f"- **{cls_name}** (confidence {conf:.2f})")
                                class_count[cls_name] = class_count.get(cls_name, 0) + 1
                        except Exception:
                            st.write("Tidak ada objek yang terdeteksi.")
                        summary = ", ".join([f"{v} {k}" for k,v in class_count.items()]) if class_count else "Tidak ada objek terdeteksi."
                        st.info(f"Ringkasan: {summary}")
                    # download button for annotated image
                    png_bytes = cv2_imencode_png(annotated_cv)
                    if png_bytes:
                        st.download_button("⬇️ Download Hasil Deteksi (PNG)", data=png_bytes, file_name="deteksi_result.png", mime="image/png")
                    st.markdown("---")
                    st.markdown("**Ringkasan singkat:**")
                    st.write("Model YOLOv8n dilatih pada dataset sepak bola (ball, goalkeeper, player, referee). Metrik: mAP & recall pada laporan.")
                else:
                    st.error("Inferensi YOLO gagal; periksa model dan file input.")
            else:
                st.error("Hasil prediksi tidak tersedia.")

# -----------------------
# PAGE: FEEDBACK
# -----------------------
elif page == "Feedback":
    st.markdown("## Feedback & Saran Pengembangan")
    st.markdown("Terima kasih sudah mencoba dashboard ini. Silakan isi form berikut untuk memberikan masukan.")

    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Nama (opsional)", placeholder="Nama kamu")
        rating = st.slider("Seberapa puas dengan penjelasan proyek ini?", 1, 5, 4)
        suggestion = st.text_area("Saran / komentar", placeholder="Tulis saran pengembangan...")
        submitted = st.form_submit_button("Kirim Saran")

        if submitted:
            entry = {
                "name": name if name else "Anonim",
                "rating": rating,
                "suggestion": suggestion,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            try:
                save_feedback(entry)
                st.success("Terima kasih! Saranmu sudah tersimpan.")
            except Exception as e:
                st.error(f"Gagal menyimpan saran: {e}")

    st.markdown("---")
    st.markdown("### 3 Saran Terbaru")
    feedbacks = read_feedback()
    # show last 3
    last3 = feedbacks[-3:][::-1] if feedbacks else []
    if last3:
        for fb in last3:
            st.markdown(f"**{fb['name']}** ({fb['time']}) — Rating: {fb['rating']}/5")
            st.write(fb["suggestion"])
            st.markdown("---")
    else:
        st.info("Belum ada saran yang masuk. Jadilah yang pertama!")

# -----------------------
# PAGE: TENTANG
# -----------------------
elif page == "Tentang":
    st.markdown("## Tentang & Kontak")
    st.markdown("**Proyek:** Ujian Tengah Semester - Praktikum Big Data")
    st.markdown("**Pembuat:** Yeni Ckrisdayanti Manalu (Mahasiswa Statistika 2022)")
    st.markdown("- Model klasifikasi: custom CNN (.h5)\n- Model deteksi: YOLOv8n (.pt)\n- Dataset: Chest X-ray (Kaggle), Football Players Detection (Kaggle/Roboflow)")
    st.markdown("---")
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(None, width=120)
    with col2:
        st.markdown("**Link & Kontak**")
        st.markdown("- GitHub: (masukkan repo kamu)\n- LinkedIn: (masukkan link)\n- Instagram: (opsional)")
    st.markdown("---")
    st.markdown("Terima kasih telah menggunakan dashboard ini. Semoga membantu presentasi dan laporan UTS kamu! 🎓")

# -----------------------
# END
# -----------------------
