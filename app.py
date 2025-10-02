import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import shap
import matplotlib.pyplot as plt
import base64

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Beban Listrik PLN", page_icon="⚡", layout="wide")

# Fungsi untuk memuat model dan explainer agar tidak di-load ulang setiap interaksi
@st.cache_resource
def load_resources():
    """Memuat model machine learning dan SHAP explainer dari file."""
    model = joblib.load('model_prediksi_beban.joblib')
    explainer = shap.TreeExplainer(model)
    return model, explainer

# Fungsi untuk mengubah file gambar menjadi format Base64 agar bisa disisipkan di HTML/CSS
def get_image_as_base64(file):
    """Membaca file gambar dan meng-encode-nya ke Base64."""
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Memuat model dan gambar logo saat aplikasi pertama kali dijalankan
model, explainer = load_resources()
logo_base64 = get_image_as_base64("logo-pln.png")

# --- CSS Kustom untuk Styling Halaman ---
st.markdown("""
<style>
/* Card untuk hasil prediksi */
.prediction-card {
    border: 2px solid #DDDDDD;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    margin: 10px;
}
.prediction-card h3 {
    margin-bottom: 10px;
    font-weight: normal;
    font-size: 1.25em;
}
.prediction-card p {
    font-size: 2em;
    font-weight: bold;
    color: #0073B4;
    margin: 0;
}

/* Header aplikasi dengan logo */
.header-container {
    background-color: #E0F7FF;
    border-radius: 15px;
    padding: 20px 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 30px;
}
.header-container img {
    width: 100px;
    margin-right: 20px;
}
.header-container h1 {
    font-size: 2.5em;
    margin: 0;
}

/* Styling untuk komponen input */
.input-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
    margin-bottom: 20px;
}
div[data-baseweb="slider"] > div > div { background: #0073B4 !important; }
div.stButton > button { background-color: #0073B4; color: white; border-radius: 10px; }
div[data-baseweb="input"] > div { background-color: #FFEB3B; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Tampilan Utama Aplikasi ---

# Menampilkan header dengan logo dan judul
st.markdown(f"""
<div class="header-container">
    <img src="data:image/png;base64,{logo_base64}" alt="PLN Logo">
    <h1>Aplikasi Prediksi Beban Listrik PLN</h1>
</div>
""", unsafe_allow_html=True)

# Area untuk input pengguna
st.header('Masukkan Waktu Prediksi')
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    d = st.date_input("Pilih Tanggal", datetime.now())
    h = st.slider("Pilih Jam", 0, 23, 10)
    st.markdown('</div>', unsafe_allow_html=True)

# Tombol prediksi ditempatkan di tengah
st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
if st.button('Buat Prediksi'):
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Menggabungkan tanggal dan jam dari input
    selected_datetime = datetime(d.year, d.month, d.day, h)
    
    # Membuat DataFrame dari input untuk diproses oleh model
    input_data = pd.DataFrame([{
        'hour': h,
        'dayofweek': d.weekday(),
        'quarter': pd.Timestamp(d).quarter,
        'month': d.month,
        'year': d.year,
        'dayofyear': d.timetuple().tm_yday
    }])
    
    # Melakukan prediksi beban listrik
    prediksi = model.predict(input_data)
    
    st.markdown("---")

    # --- Area Tampilan Hasil Prediksi ---
    res1, res2 = st.columns(2)
    
    # Kolom kiri untuk menampilkan angka prediksi
    with res1:
        st.markdown(f'<div class="prediction-card"><h3>Hasil Prediksi</h3><p>{prediksi[0]:.2f} MW</p></div>', unsafe_allow_html=True)
        
    # Kolom kanan untuk penjelasan grafik SHAP
    with res2:
        with st.expander("Lihat Cara Membaca Grafik"):
            st.info("""
            Grafik di bawah menunjukkan bagaimana setiap fitur mendorong hasil prediksi dari nilai dasarnya.
            - **Fitur berwarna merah** menaikkan nilai prediksi.
            - **Fitur berwarna biru** menurunkan nilai prediksi.
            """)
    
    # Menampilkan grafik SHAP force plot
    shap_values = explainer.shap_values(input_data)
    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        input_data.iloc[0], 
        matplotlib=True, 
        show=False, 
        text_rotation=0
    )
    fig = plt.gcf()
    fig.set_figwidth(12)
    fig.set_figheight(4)
    fig.tight_layout()
    st.pyplot(fig)
    
else:
    # Penutup div jika tombol tidak ditekan, agar layout tetap rapi
    st.markdown('</div>', unsafe_allow_html=True)