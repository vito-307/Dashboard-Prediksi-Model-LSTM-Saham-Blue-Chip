import streamlit as st
from PIL import Image
import os

# --- KONFIGURASI HALAMAN ---
# Menggunakan path relatif untuk ikon, dengan fallback
try:
    # Asumsi Anda punya logo di folder assets
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', '3.png')
    page_icon_to_use = Image.open(icon_path)
except FileNotFoundError:
    page_icon_to_use = "ℹ️" # Emoji fallback jika ikon tidak ditemukan

# Atur konfigurasi untuk halaman ini
st.set_page_config(
    page_title="Tentang Aplikasi - Blue Stock Predict",
    page_icon="ℹ️", # Bisa gunakan emoji atau path ke logo
    layout="wide"
)

st.title("ℹ️ Tentang Aplikasi Prediksi Harga 5 Emiten Saham Blue Chip")

st.markdown("---")

st.markdown("""
Aplikasi ini merupakan implementasi dari proyek penelitian tugas akhir semester yang bertujuan untuk mengembangkan sistem prediksi harga saham harian menggunakan metode _deep learning_ dan menyajikannya dalam sebuah _dashboard_ web yang interaktif dan mudah digunakan.
""")

st.header("Tujuan Aplikasi")
st.write(
    """
    - **Memberikan Prediksi:** Menyediakan prediksi harga penutupan harian untuk lima emiten saham _blue chip_ terkemuka di Bursa Efek Indonesia (BEI).
    - **Visualisasi Data:** Menampilkan data historis harga saham dalam bentuk tabel dan grafik interaktif untuk mempermudah analisis.
    - **Antarmuka Interaktif:** Memungkinkan pengguna untuk memilih emiten dan tanggal prediksi yang diinginkan di masa depan.
    - **Aksesibilitas:** Menyediakan alat bantu analisis teknis yang dapat diakses dengan mudah melalui web browser.
    """
)

st.header("Metodologi")
st.markdown(
    """
    - **Data**: Data historis harga saham (`Open`, `High`, `Low`, `Close`, `Volume`) diambil secara otomatis dari **Yahoo Finance** menggunakan library `yfinance`.
    - **Model**: Prediksi dilakukan menggunakan model _deep learning_ **Long Short-Term Memory (LSTM)**, yang sangat cocok untuk data sekuensial dan _time series_ seperti harga saham. Model ini dilatih secara terpisah untuk setiap emiten guna menangkap karakteristik unik masing-masing saham.
    - **Fitur Prediksi**: Model menggunakan harga penutupan ('Close') dari **25 hari sebelumnya** (`window_size=25`) untuk memprediksi harga penutupan di hari berikutnya. Prediksi untuk beberapa hari ke depan dilakukan secara iteratif.
    - **Evaluasi**: Performa model diukur menggunakan metrik **Mean Absolute Percentage Error (MAPE)**. Berdasarkan pengujian, model ini memiliki rata-rata performa yang akurat (MAPE ≈ 2.14%) untuk tiap-tiap model pada emiten.
    - **Framework Aplikasi**: Dashboard ini dibangun menggunakan **Streamlit**, sebuah _framework_ Python _open-source_ untuk membangun dan berbagi aplikasi data.
    """
)

st.header("Emiten yang Dicakup")
st.code("""
- BBCA (Bank Central Asia Tbk.)
- BBRI (Bank Rakyat Indonesia (Persero) Tbk.)
- TLKM (Telkom Indonesia (Persero) Tbk.)
- BMRI (Bank Mandiri (Persero) Tbk.)
- ASII (Astra International Tbk.)
""")

st.header("Disclaimer")
st.warning(
    """
    **PENTING:** Prediksi yang dihasilkan oleh aplikasi ini adalah hasil dari model matematis dan tidak boleh dianggap sebagai saran finansial atau investasi. Semua keputusan investasi tetap menjadi tanggung jawab penuh pengguna. Selalu lakukan riset Anda sendiri (_do your own research_).
    """
)

# --- BAGIAN FOTO DAN INFO PEMBUAT DENGAN DEBUGGING TAMBAHAN ---
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; font-size: 25px;'>Dibuat oleh:</h1>", unsafe_allow_html=True)
col1_1, col1_2, col1_3 = st.columns([1, 5, 1])
with col1_2:
    st.image("assets/7.jpeg",
             caption="Vito Faza Alfarizzy",
             use_container_width=True)
st.markdown("---")
st.write("**Mata Kuliah: Pengembangan Machine Learning**")
st.write("**Program Studi: Sains Data**")
st.write("**Fakultas: Teknologi Informasi**")
st.write("**Institusi: Universitas Nusa Mandiri**")
# --- AKHIR BAGIAN FOTO DAN INFO PEMBUAT ---