import streamlit as st

# ------------------------------
# Custom CSS Tema Hitam-Biru
# ------------------------------
st.markdown(
    """
    <style>
    /* Background dan teks */
    .stApp { background-color: #000000; color: #b3e5fc; font-family: 'Inter', sans-serif;}
    h1, h2, h3, h4, h5, h6 {color:#00b0ff;}
    .stButton>button {background: linear-gradient(90deg,#0288d1,#00b0ff,#4fc3f7); color:white; border-radius:10px;}
    .stButton>button:hover {transform: scale(1.02);}
    hr {border: 1px solid #00b0ff;}
    a {color: #00b0ff; text-decoration: none;}
    </style>
    """, unsafe_allow_html=True
)

# ------------------------------
# Menu Sidebar
# ------------------------------
st.sidebar.title("Pilih Fitur")
menu = st.sidebar.radio("Menu", [
    "Halaman Utama", 
    "Klasifikasi Gambar", 
    "Deteksi Objek",
    "Prediksi", 
    "Feedback",
    "Tentang Penyusun"
])

# ------------------------------
# Halaman Utama
# ------------------------------
if menu == "Halaman Utama":
    st.title("Dashboard UTS Big Data")
    st.markdown("Haii salam kenal semuanya! Aku **Yeni Ckrisdayanti Manalu**, angkatan 22. Terima kasih sudah mengunjungi dashboard UTS Big Data-ku, semoga bermanfaat. 😊")
    st.markdown("---")
    
    st.subheader("Fitur Dashboard")
    st.markdown("""
    - **Klasifikasi Gambar:** Analisis dan prediksi gambar X-ray untuk mendeteksi pneumonia.  
    - **Deteksi Objek:** Mengenali pemain, bola, wasit, dan penjaga gawang dalam dataset sepak bola.  
    - **Prediksi:** Jalankan model dan lihat hasil prediksi secara real-time.  
    - **Feedback:** Kirim masukan agar dashboard lebih baik.  
    - **Tentang Penyusun:** Info tentang pembuat dan link sosial media.
    """)
    
    st.markdown("### Kamu penasaran untuk mendalaminya? Pilih salah satu:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Klasifikasi Gambar"):
            st.session_state['page'] = 'klasifikasi_gambar'
    with col2:
        if st.button("Deteksi Objek"):
            st.session_state['page'] = 'deteksi_objek'

# ------------------------------
# Halaman Klasifikasi Gambar
# ------------------------------
elif menu == "Klasifikasi Gambar":
    st.title("Klasifikasi Gambar")
    st.markdown("Di halaman ini, kamu dapat mencoba model klasifikasi gambar X-ray untuk mendeteksi pneumonia.")
    
    st.markdown("""
    **Ringkasan Proyek:**
    - Dataset: Chest X-ray Images (Normal dan Pneumonia)
    - Total gambar: 5.856 (Normal: 1.583, Pneumonia: 4.273)
    - Preprocessing: Rescaling, Augmentasi, Resize 128x128, Grayscale
    - Model: CNN dengan beberapa Conv2D, MaxPooling, Flatten, Dense, Dropout
    - Evaluasi: Accuracy 93%, Precision & Recall ≥ 60% untuk kelas minor
    """)
    
    if st.button("Mulai Menggunakan Model"):
        st.session_state['page'] = 'prediksi'

# ------------------------------
# Halaman Deteksi Objek
# ------------------------------
elif menu == "Deteksi Objek":
    st.title("Deteksi Objek")
    st.markdown("Dataset: Football Players Detection Dataset")
    st.markdown("""
    - Jumlah gambar: 312 (Train: 251, Valid: 43, Test: 18)  
    - Kelas: ball, goalkeeper, player, referee  
    - Model: YOLOv8n dari awal (non-pretrained)  
    - Output: Bounding box, class label, confidence score
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
    st.title("Prediksi")
    st.markdown("Di sini kamu bisa menjalankan model klasifikasi atau deteksi objek secara real-time.")
    st.warning("Fitur ini sementara masih placeholder. Nanti bisa ditambahkan upload gambar dan prediksi model.")

# ------------------------------
# Halaman Feedback
# ------------------------------
elif menu == "Feedback":
    st.title("Feedback")
    st.markdown("Kirim masukanmu agar dashboard ini lebih interaktif dan menarik!")
    feedback = st.text_area("Tulis pesanmu:")
    if st.button("Kirim Feedback"):
        st.success("Terima kasih atas feedbackmu!")

# ------------------------------
# Halaman Tentang Penyusun
# ------------------------------
elif menu == "Tentang Penyusun":
    st.title("Tentang Penyusun")
    st.markdown("""
    Haii salam kenal! Aku **Yeni Ckrisdayanti Manalu**, angkatan 22.  
    Sebelumnya terima kasih sudah mengunjungi dashboard ku.  
    Ini adalah proyek UTS Big Data, semoga bermanfaat ya 😊  
    """)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 14px;'>
        © 2025 <b>Yeni Ckrisdayanti Manalu</b><br>
        Dengan bimbingan Asisten Praktikum: Bg Diaz & Bg Mus<br>
        <a href='https://www.linkedin.com/in/yeni-ckrisdayanti-manalu-741a572a9/' target='_blank'>
        🔗 LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# Session state untuk navigasi tombol
# ------------------------------
if 'page' in st.session_state:
    if st.session_state['page'] == 'klasifikasi_gambar':
        st.experimental_rerun()
    elif st.session_state['page'] == 'deteksi_objek':
        st.experimental_rerun()
    elif st.session_state['page'] == 'prediksi':
        st.experimental_rerun()
