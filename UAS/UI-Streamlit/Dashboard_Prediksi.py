import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import datetime
from tensorflow.keras.models import load_model
import os 
import traceback
from PIL import Image 
import matplotlib.pyplot as plt 
import seaborn as sns 

# --- KONFIGURASI HALAMAN DAN LOGO ---
LOGO_IMAGE_PATH = "assets/3.png" # Path ke logo Anda

try:
    # Menggunakan path relatif terhadap skrip untuk logo
    page_icon_to_use = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), LOGO_IMAGE_PATH))
except FileNotFoundError:
    page_icon_to_use = "📈" # Fallback ke emoji jika logo tidak ditemukan
except Exception as e: 
    page_icon_to_use = "⚠️" # Fallback jika ada error lain saat membuka gambar

# set_page_config harus menjadi perintah Streamlit pertama yang dipanggil
st.set_page_config(
    page_title="Blue Stock Predict", 
    page_icon=page_icon_to_use, 
    layout="wide"
)

# --- KONFIGURASI APLIKASI ---
# 1. Nama folder tempat Anda menyimpan file model dan scaler.
ARTIFACTS_DIR_NAME = "streamlit_deployment_artifacts" # Sesuai dengan kode Anda

# 2. Ukuran window (time_step) yang digunakan saat training model LSTM Anda.
WINDOW_SIZE = 25 # Sesuai dengan konfigurasi model Anda
# --- AKHIR KONFIGURASI ---

# --- PENGATURAN PATH DAN DICTIONARY EMITEN ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR_PATH = os.path.join(SCRIPT_DIR, ARTIFACTS_DIR_NAME)

emiten_dict = {
    "BBCA": {"ticker": "BBCA.JK", "model_file": os.path.join(ARTIFACTS_DIR_PATH, "BBCA.JK_model.keras"), "scaler_file": os.path.join(ARTIFACTS_DIR_PATH, "BBCA.JK_scaler.joblib")},
    "BBRI": {"ticker": "BBRI.JK", "model_file": os.path.join(ARTIFACTS_DIR_PATH, "BBRI.JK_model.keras"), "scaler_file": os.path.join(ARTIFACTS_DIR_PATH, "BBRI.JK_scaler.joblib")},
    "TLKM": {"ticker": "TLKM.JK", "model_file": os.path.join(ARTIFACTS_DIR_PATH, "TLKM.JK_model.keras"), "scaler_file": os.path.join(ARTIFACTS_DIR_PATH, "TLKM.JK_scaler.joblib")},
    "BMRI": {"ticker": "BMRI.JK", "model_file": os.path.join(ARTIFACTS_DIR_PATH, "BMRI.JK_model.keras"), "scaler_file": os.path.join(ARTIFACTS_DIR_PATH, "BMRI.JK_scaler.joblib")},
    "ASII": {"ticker": "ASII.JK", "model_file": os.path.join(ARTIFACTS_DIR_PATH, "ASII.JK_model.keras"), "scaler_file": os.path.join(ARTIFACTS_DIR_PATH, "ASII.JK_scaler.joblib")}
}


# --- FUNGSI-FUNGSI ---

@st.cache_resource
def load_prediction_assets(_model_path, _scaler_path):
    """Memuat model Keras dan scaler dari file."""
    try:
        model_obj = load_model(_model_path, compile=False)
        scaler_obj = joblib.load(_scaler_path)
        return model_obj, scaler_obj
    except Exception as e:
        raise 

def predict_future_price(model, scaler, current_historical_close_values, num_steps_to_predict, window_size=WINDOW_SIZE):
    """Memprediksi harga penutupan untuk N langkah ke depan secara iteratif."""
    if len(current_historical_close_values) < window_size:
        return None 
    last_window_data = current_historical_close_values[-window_size:].copy() 
    if last_window_data.ndim == 1: 
        last_window_data = last_window_data.reshape(-1, 1) 
    scaled_window = scaler.transform(last_window_data) 
    predicted_price_for_target_date_unscaled = None
    for i in range(num_steps_to_predict):
        X_predict = np.expand_dims(scaled_window, axis=0) 
        scaled_pred_single_step = model.predict(X_predict, verbose=0)[0][0]
        if i == num_steps_to_predict - 1: 
            predicted_price_for_target_date_unscaled = scaler.inverse_transform(np.array([[scaled_pred_single_step]]))[0][0]
        new_scaled_value_for_window = np.array([[scaled_pred_single_step]]) 
        scaled_window = np.vstack((scaled_window[1:], new_scaled_value_for_window))
    return predicted_price_for_target_date_unscaled


# --- KONTEN HALAMAN UTAMA (DASHBOARD PREDIKSI) ---

st.header("📈 Aplikasi Prediksi Harga Penutupan Harian 5 Emiten Saham Blue Chip") 
st.markdown(f"Prediksi model ini dianggap Akurat dengan nilai rata-rata Mean Absolute Percentage Error tiap-tiap emiten ialah 2.14%") 
st.markdown(f"Prediksi harga penutupan saham untuk emiten yang terdaftar. Menggunakan data historis dengan window **{WINDOW_SIZE}** hari.")

# Menampilkan informasi versi library di sidebar
try:
    import tensorflow as tf
    st.sidebar.info(f"Versi TensorFlow: {tf.__version__}")
    st.sidebar.info(f"Versi NumPy: {np.__version__}")
    st.sidebar.info(f"Versi Pandas: {pd.__version__}")
    st.sidebar.info(f"Versi yfinance: {yf.__version__}")
    st.sidebar.info(f"Versi Streamlit: {st.__version__}")
    st.sidebar.info(f"Versi Matplotlib: {plt.matplotlib.__version__}")
    st.sidebar.info(f"Versi Seaborn: {sns.__version__}") 
except ImportError as ie:
    st.sidebar.error(f"Gagal mengimpor library dasar: {ie}")
except AttributeError: 
    st.sidebar.warning("Versi library tertentu belum bisa ditampilkan.")

# Menampilkan pesan warning jika logo tidak ditemukan (setelah st.set_page_config)
if isinstance(page_icon_to_use, str): 
    if page_icon_to_use == "📈":
        st.sidebar.warning(f"File logo ('{LOGO_IMAGE_PATH}') tidak ditemukan.")
    elif page_icon_to_use == "⚠️":
        st.sidebar.error(f"Terjadi error saat mencoba memuat logo ('{LOGO_IMAGE_PATH}').")


selected_emiten_key = st.selectbox("Pilih Emiten", list(emiten_dict.keys()))

if selected_emiten_key:
    emiten_info = emiten_dict[selected_emiten_key]
    stock_ticker_symbol = emiten_info["ticker"] 
    model_file_path = emiten_info["model_file"] 
    scaler_file_path = emiten_info["scaler_file"]

    st.subheader(f"Analisis Untuk {selected_emiten_key}")

    try:
        model, scaler = load_prediction_assets(model_file_path, scaler_file_path)
        st.success(f"Model dan Scaler untuk {selected_emiten_key} berhasil dimuat.")

        today_date = datetime.date.today()
        start_date_download = today_date - datetime.timedelta(days=max(365, WINDOW_SIZE + 100)) 

        historical_data_df = yf.download(stock_ticker_symbol, start=start_date_download, end=today_date, progress=False)
        
        if historical_data_df.empty:
            st.error(f"❌ Data tidak tersedia dari Yahoo Finance untuk {stock_ticker_symbol} pada rentang tanggal yang diminta.")
        else:
            # Ambil data 'Close' yang valid (non-NaN)
            valid_close_prices = historical_data_df['Close'].dropna()

            # Menampilkan metrik harga terakhir
            if len(valid_close_prices) >= 2:
                # --- PERBAIKAN DI SINI UNTUK MENGATASI TypeError ---
                # Ambil nilai sebagai numpy array
                close_values = valid_close_prices.values
                # Ambil nilai terakhir dan sebelumnya sebagai angka tunggal (skalar)
                last_price = float(close_values[-1])
                previous_price = float(close_values[-2])
                
                delta_price = last_price - previous_price
                
                st.metric(
                    label=f"Harga Penutupan Terakhir ({valid_close_prices.index[-1].strftime('%d %b %Y')})",
                    value=f"Rp {last_price:,.2f}",
                    delta=f"Rp {delta_price:,.2f}"
                )
            
            st.subheader("Data Historis Mentah 5 Baris Terakhir")
            st.dataframe(historical_data_df.tail())

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
            
            if df_close_column is not None and not df_close_column.empty:
                last_available_data_date = df_close_column.index[-1].date()
                
                default_prediction_date = last_available_data_date + datetime.timedelta(days=1)
                target_prediction_date = st.date_input(
                    "Pilih Tanggal Prediksi",
                    value=default_prediction_date,
                    min_value=default_prediction_date, 
                    help=f"Pilih tanggal di masa depan (setelah {last_available_data_date.strftime('%Y-%m-%d')}) untuk prediksi harga penutupan."
                )

                if len(df_close_column) >= WINDOW_SIZE:
                    st.subheader(f"Grafik Harga Penutupan Historis {selected_emiten_key}")
                    
                    if not df_close_column.empty and 'Close' in df_close_column.columns:
                        try:
                            sns.set_style("whitegrid") 
                            fig, ax = plt.subplots(figsize=(10, 6)) 
                            sns.lineplot(
                                data=df_close_column['Close'], 
                                ax=ax, 
                                linewidth=1.5, 
                                color='mediumseagreen'
                            )
                            ax.set_title(f"Harga Penutupan Historis {stock_ticker_symbol}", fontsize=15)
                            ax.set_xlabel("Tanggal", fontsize=12)
                            ax.set_ylabel("Harga Close (Rp)", fontsize=12)
                            handles, labels = ax.get_legend_handles_labels()
                            if handles: 
                                ax.legend() 
                            fig.autofmt_xdate() 
                            plt.tight_layout() 
                            st.pyplot(fig)
                            plt.close(fig) 
                        except Exception as e_sns:
                            st.warning(f"Tidak dapat menampilkan grafik Seaborn: {e_sns}")
                            st.text(traceback.format_exc())
                    else:
                        st.warning("Tidak dapat menampilkan grafik karena data 'Close' tidak valid atau kosong setelah diproses.")

                    num_steps_to_predict = (target_prediction_date - last_available_data_date).days

                    if num_steps_to_predict > 0:
                        close_values_for_prediction_base = df_close_column.values
                        predicted_price = predict_future_price(model, scaler, 
                                                              close_values_for_prediction_base, 
                                                              num_steps_to_predict, 
                                                              window_size=WINDOW_SIZE)

                        if predicted_price is not None:
                            st.success(f"📊 Prediksi harga penutupan untuk {selected_emiten_key} pada tanggal **{target_prediction_date.strftime('%Y-%m-%d')}**: **Rp {predicted_price:,.2f}**")
                        else:
                            st.warning(f"⚠️ Prediksi tidak dapat dibuat.")
                    elif num_steps_to_predict == 0: 
                        st.warning("Tanggal prediksi adalah tanggal data terakhir. Tidak ada prediksi ke depan.")
                    else: 
                        st.error("Tanggal prediksi yang dipilih adalah sebelum data historis terakhir.")
                else: 
                    st.warning(f"⚠️ Data historis 'Close' yang valid ({len(df_close_column)}) kurang dari `WINDOW_SIZE` ({WINDOW_SIZE}). Grafik dan prediksi tidak dapat ditampilkan.")
                    if not df_close_column.empty: 
                        st.subheader(f"Data Harga Penutupan (Clean) yang Tersedia ({selected_emiten_key})")
                        st.dataframe(df_close_column)
            else: 
                 st.warning("Tidak ada data 'Close' yang valid untuk diproses.")

    except FileNotFoundError as fnf_error:
        st.error(f"❌ FILE TIDAK DITEMUKAN. Pastikan nama folder di `ARTIFACTS_DIR_NAME` ('{ARTIFACTS_DIR_NAME}') sudah benar, folder tersebut berada di direktori yang sama dengan skrip ini ('{SCRIPT_DIR}'), dan berisi file yang dibutuhkan.")
        st.error(f"   Path Model Absolut yang dicari: {model_file_path}") 
        st.error(f"   Path Scaler Absolut yang dicari: {scaler_file_path}")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan yang tidak terduga pada aplikasi: {e}")
        st.text("LOKASI ERROR (TRACEBACK LENGKAP):")
        st.text(traceback.format_exc()) 
else:
    st.info("Silakan pilih emiten untuk memulai prediksi.")

