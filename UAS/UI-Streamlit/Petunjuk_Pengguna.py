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
    page_icon_to_use = "❓" # Emoji fallback jika ikon tidak ditemukan

st.set_page_config(
    page_title="Petunjuk Pengguna - Blue Stock Predict",
    page_icon=page_icon_to_use,
    layout="wide"
)

# --- KONTEN HALAMAN ---

st.title("❓ Petunjuk Pengguna")
st.markdown("---")
st.markdown("Selamat datang di Prediksi Harga Penutupan Harian 5 Emiten Saham Blue Chip! Berikut adalah panduan singkat tentang cara menggunakan dashboard ini.")

# --- CARA MENGGUNAKAN DASHBOARD ---
st.header("Cara Menggunakan Dashboard Prediksi")

# 1. Pilih Emiten
st.subheader("1. Pilih Emiten Saham")
st.info("""
Pada menu *dropdown* di bagian atas halaman **Dashboard Prediksi**, pilih salah satu dari lima emiten saham _blue chip_ yang tersedia. Aplikasi akan secara otomatis memuat model prediksi yang sesuai dan data historis untuk emiten yang Anda pilih.
""")
# Contoh gambar ilustrasi
st.image("assets/5.png",
         caption="Lokasi menu dropdown untuk memilih emiten.")

# 2. Lihat Data Historis
st.subheader("2. Analisis Data Historis")
st.info("""
Setelah memilih emiten, aplikasi akan menampilkan:
- **Data Harga Penutupan Terakhir:** Harga penutupan terakhir hari sebelumnya dari emiten yang dipilih yang diunduh langsung dari Yahoo Finance.        
- **Tabel Data Mentah:** Data historis yang diunduh langsung dari Yahoo Finance.
- **Tabel Data Harga Penutupan (Clean):** Data harga penutupan yang sudah dibersihkan dan siap digunakan untuk grafik dan prediksi.
- **Grafik Harga Penutupan:** Visualisasi pergerakan harga penutupan saham selama periode waktu terakhir untuk membantu Anda melihat tren.        
""")

# 3. Pilih Tanggal Prediksi
st.subheader("3. Pilih Tanggal Prediksi")
st.info("""
Gunakan widget kalender untuk memilih tanggal di masa depan yang ingin Anda prediksi.
- **Tanggal Default:** Secara otomatis diatur ke hari berikutnya setelah data historis terakhir yang tersedia.
- **Batasan:** Anda hanya bisa memilih tanggal di masa depan.
""")
st.image("assets/6.png",
         caption="Widget kalender untuk memilih tanggal.")

# 4. Lihat Hasil Prediksi
st.subheader("4. Lihat Hasil Prediksi")
st.success("""
Setelah tanggal dipilih, aplikasi akan secara otomatis menghitung dan menampilkan prediksi harga penutupan untuk tanggal tersebut. Hasilnya akan ditampilkan dalam sebuah kotak hijau di bagian bawah halaman.
""")

# --- INFORMASI TAMBAHAN ---
st.markdown("---")
st.header("Informasi Tambahan")

with st.expander("Klik untuk melihat detail Metodologi"):
    st.markdown(
        """
        - **Model**: Prediksi dilakukan menggunakan model _deep learning_ **Long Short-Term Memory (LSTM)**, yang sangat cocok untuk data sekuensial dan _time series_ seperti harga saham.
        - **Data**: Data historis diambil secara _real-time_ dari **Yahoo Finance**.
        - **Fitur Prediksi**: Model menggunakan harga penutupan ('Close') dari **25 hari sebelumnya** untuk memprediksi harga di hari berikutnya.
        - **Prediksi Jangka Panjang**: Prediksi untuk beberapa hari ke depan dilakukan secara **iteratif**, di mana hasil prediksi hari ini digunakan sebagai masukan untuk memprediksi hari berikutnya. Perlu diingat, akurasi cenderung menurun semakin jauh prediksi dilakukan. Karena model bekerja optimal untuk melakukan prediksi harian.
        """
    )

with st.expander("Klik untuk melihat Disclaimer"):
    st.warning(
        """
        **PENTING:** Prediksi yang dihasilkan oleh aplikasi ini adalah hasil dari model matematis dan tidak boleh dianggap sebagai saran finansial atau investasi. Semua keputusan investasi tetap menjadi tanggung jawab penuh pengguna. Selalu lakukan riset Anda sendiri (_do your own research_).
        """
    )

st.markdown("---")
st.write("Semoga berhasil dan selamat menganalisis!")
