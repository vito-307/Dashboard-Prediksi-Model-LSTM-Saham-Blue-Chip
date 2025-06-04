import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import datetime
from tensorflow.keras.models import load_model
import os 
import traceback

st.set_page_config(page_title="Prediksi Harga Saham", layout="wide")

# --- KONFIGURASI PENTING YANG PERLU ANDA SESUAIKAN ---
# 1. Nama folder tempat Anda menyimpan file model dan scaler.
#    Folder ini diasumsikan berada di lokasi yang sama dengan skrip app.py ini.
ARTIFACTS_DIR_NAME = "streamlit_deployment_artifacts" # GANTI JIKA NAMA FOLDER ANDA BERBEDA

# 2. Ukuran window (time_step) yang digunakan saat training model LSTM Anda.
#    Ini HARUS sesuai dengan model Anda (misalnya 25 jika batch_shape model [None, 25, 1]).
WINDOW_SIZE = 25
# --- AKHIR KONFIGURASI PENTING ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR_PATH = os.path.join(SCRIPT_DIR, ARTIFACTS_DIR_NAME)

emiten_dict = {
    "BBCA": {"ticker": "BBCA.JK", "model_file": os.path.join(ARTIFACTS_DIR_PATH, "BBCA.JK_model.keras"), "scaler_file": os.path.join(ARTIFACTS_DIR_PATH, "BBCA.JK_scaler.joblib")},
    "BBRI": {"ticker": "BBRI.JK", "model_file": os.path.join(ARTIFACTS_DIR_PATH, "BBRI.JK_model.keras"), "scaler_file": os.path.join(ARTIFACTS_DIR_PATH, "BBRI.JK_scaler.joblib")},
    "TLKM": {"ticker": "TLKM.JK", "model_file": os.path.join(ARTIFACTS_DIR_PATH, "TLKM.JK_model.keras"), "scaler_file": os.path.join(ARTIFACTS_DIR_PATH, "TLKM.JK_scaler.joblib")},
    "BMRI": {"ticker": "BMRI.JK", "model_file": os.path.join(ARTIFACTS_DIR_PATH, "BMRI.JK_model.keras"), "scaler_file": os.path.join(ARTIFACTS_DIR_PATH, "BMRI.JK_scaler.joblib")},
    "ASII": {"ticker": "ASII.JK", "model_file": os.path.join(ARTIFACTS_DIR_PATH, "ASII.JK_model.keras"), "scaler_file": os.path.join(ARTIFACTS_DIR_PATH, "ASII.JK_scaler.joblib")}
}

@st.cache_resource
def load_prediction_assets(_model_path, _scaler_path):
    """Memuat model Keras dan scaler dari file."""
    try:
        model_obj = load_model(_model_path, compile=False)
        scaler_obj = joblib.load(_scaler_path)
        return model_obj, scaler_obj
    except Exception as e:
        raise Exception(f"Gagal memuat aset dari '{_model_path}' atau '{_scaler_path}': {e}")

def predict_next_day_price(model, scaler, historical_close_values, window_size=WINDOW_SIZE):
    """Memprediksi harga penutupan untuk hari berikutnya."""
    if len(historical_close_values) < window_size:
        return None 
    
    last_window_data = historical_close_values[-window_size:]
    if last_window_data.ndim == 1: 
        last_window_data = last_window_data.reshape(-1, 1)
    
    scaled_last_window = scaler.transform(last_window_data)
    X_predict = np.expand_dims(scaled_last_window, axis=0) 
    
    scaled_prediction = model.predict(X_predict)[0][0]
    unscaled_prediction = scaler.inverse_transform(np.array([[scaled_prediction]]))[0][0]
    return unscaled_prediction

# --- UI Streamlit ---
st.title("📈 Aplikasi Prediksi Harga Saham")

try:
    import tensorflow as tf
    st.sidebar.info(f"Versi TensorFlow: {tf.__version__}")
    st.sidebar.info(f"Versi NumPy: {np.__version__}")
    st.sidebar.info(f"Versi Pandas: {pd.__version__}")
    st.sidebar.info(f"Versi yfinance: {yf.__version__}")
    st.sidebar.info(f"Versi Streamlit: {st.__version__}")
except ImportError as ie:
    st.sidebar.error(f"Gagal mengimpor library dasar: {ie}")

st.markdown(f"Prediksi harga penutupan saham untuk emiten yang terdaftar. Menggunakan data historis dengan window **{WINDOW_SIZE}** hari.")
selected_emiten_key = st.selectbox("Pilih Emiten", list(emiten_dict.keys()))

if selected_emiten_key:
    emiten_info = emiten_dict[selected_emiten_key]
    stock_ticker_symbol = emiten_info["ticker"] 
    model_file_path = emiten_info["model_file"] 
    scaler_file_path = emiten_info["scaler_file"]

    st.subheader(f"Prediksi untuk: {selected_emiten_key} ({stock_ticker_symbol})")

    try:
        model, scaler = load_prediction_assets(model_file_path, scaler_file_path)
        st.success(f"Model dan Scaler untuk {selected_emiten_key} berhasil dimuat.")

        end_date = datetime.date.today()
        start_date_download = end_date - datetime.timedelta(days=max(250, WINDOW_SIZE + 100)) 

        historical_data_df = yf.download(stock_ticker_symbol, start=start_date_download, end=end_date, progress=False)
        
        if historical_data_df.empty:
            st.error(f"❌ Data tidak tersedia dari Yahoo Finance untuk {stock_ticker_symbol} pada rentang tanggal yang diminta.")
        else:
            # MENAMPILKAN TABEL DATA MENTAH
            st.subheader("Sampel Data Historis Mentah (5 Baris Pertama)")
            st.dataframe(historical_data_df.head())
            # Anda bisa mengaktifkan baris di bawah jika ingin melihat 5 baris terakhir juga
            # st.subheader("Sampel Data Historis Mentah (5 Baris Terakhir)")
            # st.dataframe(historical_data_df.tail())

            df_close_column = None 
            if 'Close' in historical_data_df.columns:
                df_close_column = historical_data_df[['Close']].astype(float).dropna()
            elif isinstance(historical_data_df.columns, pd.MultiIndex) and ('Close', stock_ticker_symbol) in historical_data_df.columns:
                series_close_price = historical_data_df[('Close', stock_ticker_symbol)]
                df_close_column = series_close_price.to_frame(name='Close').astype(float).dropna()
            else:
                st.error(f"❌ Kolom 'Close' tidak ditemukan dalam data yang diunduh untuk {stock_ticker_symbol}.")
                st.write("Kolom yang tersedia:", historical_data_df.columns)
                st.stop()
            
            if df_close_column is not None and len(df_close_column) >= WINDOW_SIZE:
                # MENAMPILKAN TABEL DATA 'CLOSE' YANG SUDAH BERSIH
                st.subheader(f"Data Harga Penutupan (Clean) untuk Grafik & Prediksi ({selected_emiten_key}) - 5 Baris Pertama")
                st.dataframe(df_close_column.head())
                # Anda bisa mengaktifkan baris di bawah jika ingin melihat 5 baris terakhir juga
                # st.subheader(f"Data Harga Penutupan (Clean) untuk Grafik & Prediksi ({selected_emiten_key}) - 5 Baris Terakhir")
                # st.dataframe(df_close_column.tail())

                st.subheader(f"Grafik Harga Penutupan Historis ({selected_emiten_key})")
                if not df_close_column.empty and 'Close' in df_close_column.columns:
                    st.line_chart(df_close_column['Close']) 
                else:
                    st.warning("Tidak dapat menampilkan grafik karena data 'Close' tidak valid atau kosong setelah diproses.")

                close_values_for_prediction = df_close_column.values
                predicted_price = predict_next_day_price(model, scaler, close_values_for_prediction, window_size=WINDOW_SIZE)

                if predicted_price is not None:
                    st.success(f"📊 Prediksi harga penutupan berikutnya untuk {selected_emiten_key}: **Rp {predicted_price:,.2f}**")
                else:
                    st.warning(f"⚠️ Prediksi tidak dapat dibuat (kemungkinan karena data tidak cukup setelah diproses lebih lanjut).")
            else:
                st.warning(f"⚠️ Data historis 'Close' yang valid ({len(df_close_column) if df_close_column is not None else 0} hari) kurang dari `WINDOW_SIZE` ({WINDOW_SIZE} hari). Tabel data 'Close' (clean), grafik, dan prediksi tidak dapat ditampilkan.")
                # Tetap tampilkan data 'Close' yang ada meskipun kurang, agar pengguna bisa lihat
                if df_close_column is not None and not df_close_column.empty: 
                    st.subheader(f"Data Harga Penutupan (Clean) yang Tersedia ({selected_emiten_key}) - Meskipun Kurang dari {WINDOW_SIZE} hari")
                    st.dataframe(df_close_column)

    except FileNotFoundError as fnf_error:
        st.error(f"❌ FILE TIDAK DITEMUKAN. Pastikan nama folder di `ARTIFACTS_DIR_NAME` ('{ARTIFACTS_DIR_NAME}') sudah benar, folder tersebut berada di direktori yang sama dengan skrip ini ('{SCRIPT_DIR}'), dan berisi file model/scaler yang dibutuhkan.")
        st.error(f"   Detail Error Sistem: {fnf_error}")
        st.error(f"   Path Model Absolut yang dicari: {model_file_path}") 
        st.error(f"   Path Scaler Absolut yang dicari: {scaler_file_path}")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan yang tidak terduga: {e}")
        st.text("LOKASI ERROR (TRACEBACK LENGKAP):")
        st.text(traceback.format_exc()) 
else:
    st.info("Silakan pilih emiten untuk memulai prediksi.")