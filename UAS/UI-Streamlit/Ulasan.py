import streamlit as st
from PIL import Image
import os
import datetime
import gspread
from google.oauth2.service_account import Credentials
import traceback

# --- KONFIGURASI HALAMAN ---
try:
    # Menggunakan path relatif untuk menemukan logo dari folder pages
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', '3.png')
    page_icon_to_use = Image.open(icon_path)
except FileNotFoundError:
    page_icon_to_use = "✉️" # Emoji fallback jika ikon tidak ditemukan

st.set_page_config(
    page_title="Masukan & Saran - Blue Stock Predict",
    page_icon=page_icon_to_use,
    layout="centered" # Menggunakan layout 'centered' agar form terlihat lebih fokus
)

# --- FUNGSI UNTUK KONEKSI GOOGLE SHEETS ---

# Menggunakan cache_resource agar koneksi tidak dibuat ulang setiap kali ada interaksi
@st.cache_resource
def get_gspread_client():
    """Membuat koneksi ke Google Sheets menggunakan kredensial dari Streamlit Secrets."""
    try:
        # Mengambil kredensial dari st.secrets
        # Pastikan Anda sudah membuat file .streamlit/secrets.toml
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"],
        )
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        # Menampilkan pesan error yang lebih informatif jika koneksi gagal
        st.error(f"Gagal terhubung ke Google Sheets.")
        st.warning("Pastikan Anda sudah mengikuti panduan untuk mengatur file '.streamlit/secrets.toml' dengan benar.")
        st.code(f"Detail Error: {e}\n{traceback.format_exc()}")
        return None

def save_feedback_to_gsheet(client, sheet_name, worksheet_name, data_row):
    """Menyimpan satu baris data ke worksheet di Google Sheet."""
    if client is None:
        return False, "Koneksi Google Sheets tidak tersedia."
    try:
        # Buka spreadsheet berdasarkan nama
        spreadsheet = client.open(sheet_name)
        # Buka worksheet (tab) berdasarkan nama
        worksheet = spreadsheet.worksheet(worksheet_name)
        # Tambahkan baris baru di paling bawah
        worksheet.append_row(data_row, value_input_option='USER_ENTERED')
        return True, "Data berhasil disimpan."
    except gspread.exceptions.SpreadsheetNotFound:
        return False, f"Spreadsheet dengan nama '{sheet_name}' tidak ditemukan di akun Google Anda."
    except gspread.exceptions.WorksheetNotFound:
        return False, f"Worksheet dengan nama '{worksheet_name}' tidak ditemukan di dalam spreadsheet '{sheet_name}'."
    except Exception as e:
        return False, f"Terjadi error saat menyimpan ke Google Sheet: {e}"

# --- KONTEN HALAMAN ---

st.title("✉️ Masukan dan Saran")
st.markdown("---")
st.markdown("Kami sangat menghargai masukan Anda untuk membantu kami meningkatkan kualitas aplikasi ini. Silakan isi formulir di bawah ini.")

# Menggunakan st.form untuk mengelompokkan input dan memiliki satu tombol submit
with st.form(key="feedback_form", clear_on_submit=True):
    # Input untuk nama pengguna (opsional)
    nama_pengguna = st.text_input(
        label="Nama Anda (Opsional)",
        placeholder="Masukkan nama Anda..."
    )

    # Input untuk rating menggunakan radio button
    st.write("Bagaimana penilaian Anda terhadap aplikasi ini?")
    rating = st.radio(
        "Pilih rating (1=Buruk, 5=Sangat Baik):",
        options=['🤑', '🤑🤑', '🤑🤑🤑', '🤑🤑🤑🤑', '🤑🤑🤑🤑🤑'],
        horizontal=True, # Menampilkan pilihan secara horizontal
        index=4 # Default pilihan ke 5 bintang
    )

    # Input untuk masukan atau saran dalam bentuk text area
    masukan_pengguna = st.text_area(
        label="Tuliskan masukan atau saran Anda di sini:",
        placeholder="Contoh: Aplikasi ini sangat membantu, mungkin bisa ditambahkan fitur analisis sentimen.",
        height=150, # Mengatur tinggi area teks
        max_chars=1000, # Membatasi jumlah karakter
        help="Tuliskan saran, kritik, atau laporan bug yang Anda temukan."
    )

    # Tombol untuk mengirim form
    submitted = st.form_submit_button("Kirim Masukan")

# Logika setelah form disubmit
if submitted:
    # Memeriksa apakah pengguna menulis sesuatu di kotak masukan
    if masukan_pengguna:
        # Persiapkan data untuk dikirim ke Google Sheets
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rating_value = len(rating) # Mengambil jumlah bintang sebagai nilai rating
        data_to_save = [timestamp, nama_pengguna if nama_pengguna else 'Anonim', rating_value, masukan_pengguna]

        # Inisialisasi koneksi dan simpan data
        gspread_client = get_gspread_client()
        
        # --- GANTI DENGAN NAMA GOOGLE SHEET DAN WORKSHEET ANDA ---
        nama_google_sheet = "Data Masukan Aplikasi Saham" 
        nama_worksheet = "Sheet1" # Biasanya nama tab default

        # Kirim data ke Google Sheets
        success, message = save_feedback_to_gsheet(gspread_client, nama_google_sheet, nama_worksheet, data_to_save)

        if success:
            st.success("✅ Terima kasih! Masukan Anda telah berhasil disimpan.")
            st.balloons()
        else:
            st.error(f"❌ Gagal menyimpan masukan. {message}")

    else:
        # Menampilkan pesan error jika kotak masukan kosong
        st.error("❌ Harap isi kotak masukan sebelum mengirim.")

st.markdown("---")
st.markdown("Terima kasih telah membantu kami menjadi lebih baik!")
