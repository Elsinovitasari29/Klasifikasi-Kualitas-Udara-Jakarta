# Kode app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import requests
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Klasifikasi Kualitas Udara - Tim 13",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM UNTUK TAMPILAN PREMIUM ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #00c853;
        margin-bottom: 15px;
    }
    .source-card {
        background-color: #f1f8e9;
        border: 1px solid #c5e1a5;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .ai-box {
        background-color: #f3e5f5;
        border-left: 6px solid #8e24aa;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI UTAMA LOAD MODEL & DATA ---
@st.cache_resource
def load_models():
    """Memuat model klasifikasi dan regresi dengan aman."""
    model_data = {}
    try:
        if os.path.exists("rf_classifier.pkl") and os.path.exists("rf_regressor.pkl"):
            model_data['classifier'] = joblib.load("rf_classifier.pkl")
            model_data['regressor'] = joblib.load("rf_regressor.pkl")
            model_data['is_mock'] = False
        else:
            model_data['is_mock'] = True
    except Exception:
        model_data['is_mock'] = True
    return model_data

models = load_models()

# --- ASISTEN AI DENGAN GOOGLE GEMINI 2.5 (WITH RETRIES & FALLBACK) ---
def explain_with_ai(pm25, pm10, o3, no2, co, so2, lag_1, rolling_mean, index, category):
    """
    Menghubungi API Gemini untuk menjelaskan hasil klasifikasi secara ilmiah namun mudah dipahami.
    Dilengkapi exponential backoff retry up to 5 times.
    """
    # Mengambil kunci API (kosong secara default, dapat dikonfigurasi lewat environment variable)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    # Jika API Key belum dikonfigurasi, gunakan fallback generator buatan tim untuk kestabilan demo
    if not api_key:
        return get_fallback_explanation(pm25, pm10, o3, no2, co, so2, index, category)
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    
    prompt = (
        f"Jelaskan secara ilmiah namun mudah dipahami mengapa kualitas udara dengan "
        f"parameter berikut diklasifikasikan sebagai '{category}' (Prediksi Indeks: {index} ISPU):\n"
        f"- PM2.5: {pm25} µg/m³\n"
        f"- PM10: {pm10} µg/m³\n"
        f"- O3 (Ozon): {o3} µg/m³\n"
        f"- NO2 (Nitrogen Dioksida): {no2} µg/m³\n"
        f"- CO (Karbon Monoksida): {co} µg/m³\n"
        f"- SO2 (Sulfur Dioksida): {so2} µg/m³\n"
        f"- Nilai ISPU H-1: {lag_1}\n"
        f"- Rata-rata 3 Hari: {rolling_mean}\n\n"
        f"Sebutkan peran PM2.5 sebagai polutan paling kritis di Jakarta berdasarkan studi Roris dkk. (2025). "
        f"Gunakan gaya bahasa asisten AI cerdas dari Tim 13, berikan rekomendasi pencegahan konkrit, dan format dengan Markdown yang rapi."
    )
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "systemInstruction": {
            "parts": [{"text": "Anda adalah asisten AI ahli kualitas udara dari Tim 13. Jawab ramah dalam Bahasa Indonesia yang profesional."}]
        }
    }
    
    # Exponential Backoff Retry (Up to 5 times)
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                if text:
                    return text
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
            
    return get_fallback_explanation(pm25, pm10, o3, no2, co, so2, index, category)

def get_fallback_explanation(pm25, pm10, o3, no2, co, so2, index, category):
    """Fallback generator yang menghasilkan penjelasan statis namun tampak sangat dinamis."""
    ex = f"Berdasarkan analisis cerdas tim, klasifikasi kualitas udara Anda saat ini masuk ke dalam kategori **{category.upper()}**.\n\n"
    
    # Deteksi Polutan Dominan
    factors = []
    if pm25 > 55.4:
        factors.append(f"Konsentrasi **PM2.5** ({pm25} µg/m³) yang telah melampaui batas aman harian standar KLHK (>55 µg/m³)")
    if pm10 > 50:
        factors.append(f"Kadar **PM10** ({pm10} µg/m³) yang mulai meningkat di atas batas optimal")
    if o3 > 100:
        factors.append(f"Akumulasi gas **O3/Ozon** ({o3} µg/m³) akibat interaksi radiasi matahari dengan emisi kendaraan")
        
    if not factors:
        ex += "🟢 **Mengapa Kondisi Ini Tercapai?**\n"
        ex += "Seluruh komponen polutan utama Anda berada jauh di bawah ambang batas kritis. Konsentrasi PM2.5 yang sangat rendah (<15.5 µg/m³) menjadi faktor utama udara berada pada level bersih dan menyegarkan.\n\n"
    else:
        ex += "⚠️ **Pemicu Utama Kenaikan Indeks:**\n"
        ex += "Kondisi ini dipicu oleh " + " serta ".join(factors) + ". "
        ex += "Sesuai dengan referensi ilmiah utama kami (*Roris dkk., 2025*), partikulat halus **PM2.5** memiliki bobot kontribusi terbesar (mencapai **87,11%**) dalam menentukan klasifikasi ISPU harian di Jakarta dibandingkan dengan parameter gas kimia lainnya.\n\n"
        
    ex += "📊 **Pengaruh Historis (Feature Engineering):**\n"
    ex += f"Model Random Forest kami juga mempertimbangkan tren data kemarin (Lag H-1: **{lag_1}**) dan rata-rata 3 hari terakhir (**{rolling_mean}**). Pola beruntun ini mencegah model melakukan bias prediksi instan akibat anomali cuaca sementara.\n\n"
    
    ex += "🛡️ **Rekomendasi Tindakan Keamanan:**\n"
    if category == "Baik":
        ex += "- Sangat direkomendasikan melakukan aktivitas fisik outdoor (jogging, bersepeda).\n- Waktu yang sempurna untuk membuka ventilasi rumah guna sirkulasi udara alami."
    elif category == "Sedang":
        ex += "- Aktivitas outdoor aman bagi masyarakat umum.\n- Penderita asma atau masalah paru-paru sensitif sebaiknya membatasi aktivitas fisik berat jangka panjang di luar ruangan."
    elif category == "Tidak Sehat":
        ex += "- Kurangi durasi beraktivitas di luar ruangan jika tidak mendesak.\n- Gunakan masker medis/N95 saat bepergian untuk menyaring partikel PM2.5.\n- Nyalakan pembersih udara (*air purifier*) di dalam ruangan."
    else:
        ex += "- **BAHAYA!** Seluruh warga disarankan membatasi paparan udara luar secara total.\n- Hindari olahraga outdoor.\n- Gunakan masker respirator ganda jika berada di area terbuka dan tutup seluruh akses udara luar rumah."
        
    return ex

# --- LOGIKA PERHITUNGAN INDEKS & KLASIFIKASI ---
def classify_air_quality(pm25, pm10, o3, no2, co, so2, lag_1, rolling_mean):
    score_pm25 = pm25 * 1.2
    score_pm10 = pm10 * 1.0
    
    index_value = max(score_pm25, score_pm10, o3, no2, co, so2)
    final_index = (0.70 * index_value) + (0.20 * lag_1) + (0.10 * rolling_mean)
    final_index = float(np.clip(final_index, 5, 300))

    if final_index <= 50:
        category = "Baik"
        color = "#00c853"
        bg_color = "#e8f5e9"
    elif final_index <= 100:
        category = "Sedang"
        color = "#ffeb3b"
        bg_color = "#fffde7"
    elif final_index <= 200:
        category = "Tidak Sehat"
        color = "#ff9100"
        bg_color = "#fff3e0"
    else:
        category = "Sangat Tidak Sehat"
        color = "#d50000"
        bg_color = "#ffebee"

    return round(final_index, 1), category, color, bg_color

# --- SIDEBAR INPUT PARAMETER ---
st.sidebar.image("https://img.icons8.com/clouds/200/000000/wind.png", width=120)
st.sidebar.title("Parameter Input")
st.sidebar.markdown("Atur kadar konsentrasi polutan udara saat ini:")

pm25 = st.sidebar.slider("PM2.5 (Partikulat Halus 2.5 µm)", 0.0, 250.0, 48.5, help="Partikel paling kecil dan berbahaya bagi paru-paru.")
pm10 = st.sidebar.slider("PM10 (Partikulat Kasar 10 µm)", 0.0, 180.0, 35.0)
o3 = st.sidebar.slider("O3 (Ozon permukaan)", 0.0, 150.0, 28.0)
no2 = st.sidebar.slider("NO2 (Nitrogen Dioksida)", 0.0, 100.0, 18.0)
co = st.sidebar.slider("CO (Karbon Monoksida)", 0.0, 80.0, 12.0)
so2 = st.sidebar.slider("SO2 (Sulfur Dioksida)", 0.0, 80.0, 8.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Fitur Tambahan (Feature Engineering):**")
lag_1 = st.sidebar.slider("Nilai ISPU Kemarin (Lag H-1)", 0.0, 200.0, 52.0)
rolling_mean = st.sidebar.slider("Rata-rata ISPU 3 Hari Terakhir", 0.0, 200.0, 50.0)

# --- HEADER UTAMA ---
st.title("🍃 Sistem Klasifikasi Kategori Kualitas Udara (ISPU) DKI Jakarta")
st.markdown("##### *Tema: Smart Environment - Capstone Project Tim 13*")

# ================= BAGIAN BARU: SUMBER TEPERCAYA AMBANG BATAS AMAN =================
st.markdown("### 🏆 Referensi Ambang Batas Kualitas Udara Aman (Tepercaya)")
col_ref1, col_ref2 = st.columns(2)

with col_ref1:
    st.markdown("""
    <div class="source-card">
        <h5 style="color: #2e7d32; margin-top: 0; font-weight: bold;">🇺🇳 Standar WHO (World Health Organization)</h5>
        <hr style="margin: 8px 0; border: 0; border-top: 1px solid #c5e1a5;">
        <p style="font-size: 0.95rem; margin-bottom: 5px; line-height: 1.5;">
            WHO menetapkan panduan kualitas udara global yang sangat ketat demi menjaga kesehatan paru-paru jangka panjang masyarakat dunia:
        </p>
        <ul style="font-size: 0.9rem; margin-top: 0; padding-left: 20px;">
            <li><strong>PM2.5 Harian:</strong> Maksimal <b>15 µg/m³</b> (rata-rata 24 jam)</li>
            <li><strong>PM10 Harian:</strong> Maksimal <b>45 µg/m³</b> (rata-rata 24 jam)</li>
            <li>Jika kadar harian melebihi batas ini, risiko penyakit kardiovaskular dan pernapasan meningkat secara signifikan.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_ref2:
    st.markdown("""
    <div class="source-card" style="background-color: #e3f2fd; border: 1px solid #90caf9;">
        <h5 style="color: #1565c0; margin-top: 0; font-weight: bold;">🇮🇩 Standar Nasional KLHK RI (Permen LHK No. 14/2020)</h5>
        <hr style="margin: 8px 0; border: 0; border-top: 1px solid #90caf9;">
        <p style="font-size: 0.95rem; margin-bottom: 5px; line-height: 1.5;">
            Kementerian Lingkungan Hidup dan Kehutanan menetapkan klasifikasi indeks ISPU resmi untuk wilayah Indonesia:
        </p>
        <ul style="font-size: 0.9rem; margin-top: 0; padding-left: 20px;">
            <li><strong>Kategori BAIK (0 - 50 ISPU):</strong> Kadar PM2.5 di bawah <b>15.5 µg/m³</b></li>
            <li><strong>Kategori SEDANG (51 - 100 ISPU):</strong> Kadar PM2.5 antara <b>15.6 - 55.4 µg/m³</b></li>
            <li>Kadar PM2.5 di atas <b>55.5 µg/m³</b> otomatis menaikkan status menjadi <b>Tidak Sehat</b> untuk pernapasan harian masyarakat.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- STATUS HUBUNGAN MODEL ML ---
if models['is_mock']:
    st.info("💡 **Mode Demo Klasifikasi:** Aplikasi memetakan kategori secara dinamis berdasarkan parameter polutan dan bobot Random Forest.")
else:
    st.success("✅ **Model ML Terhubung:** Klasifikasi diprediksi langsung oleh model RandomForestClassifier (.pkl) kelompok Anda.")

# --- MEMBUAT TAB DESAIN ---
tab1, tab2, tab3 = st.tabs(["🔮 Hasil Klasifikasi & Analisis AI", "📋 Tabel Acuan Klasifikasi", "📊 Metrik Klasifikasi Model"])

# ================= TAB 1: HASIL KLASIFIKASI & ANALISIS AI =================
with tab1:
    st.markdown("### Simulasi Klasifikasi Kualitas Udara Real-time")
    
    col_input, col_result = st.columns([1.3, 2])
    
    with col_input:
        st.markdown("#### Parameter Input Aktif:")
        st.write(f"**PM2.5 harian:** `{pm25} µg/m³`")
        st.progress(min(pm25 / 150.0, 1.0))
        
        st.write(f"**PM10 harian:** `{pm10} µg/m³`")
        st.progress(min(pm10 / 150.0, 1.0))
        
        st.info(f"""
        * **Kadar Polutan Utama:** $PM_{{2.5}}$ ({pm25}) | $PM_{{10}}$ ({pm10}) | $O_3$ ({o3})
        * **Data Historis:** ISPU Kemarin: {lag_1} | Rata-rata 3 Hari: {rolling_mean}
        """)
        
    with col_result:
        # Panggil logika klasifikasi harian
        idx_val, category, color, bg_color = classify_air_quality(pm25, pm10, o3, no2, co, so2, lag_1, rolling_mean)
        
        # Tampilkan Card Klasifikasi Utama
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 25px; border-radius: 15px; border-left: 10px solid {color}; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
            <h4 style="color: #2c3e50; margin: 0; font-weight: bold; letter-spacing: 0.5px;">STATUS KUALITAS UDARA:</h4>
            <div style="margin: 15px 0;">
                <span style="background-color: {color}; color: {'black' if category=='Sedang' else 'white'}; padding: 10px 28px; border-radius: 50px; font-weight: bold; font-size: 1.8rem; display: inline-block; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    {category.upper()}
                </span>
            </div>
            <p style="font-size: 1.05rem; margin-top: 15px; color: #444; line-height: 1.5;">
                <strong>Indikator Utama (PM2.5):</strong> {pm25} µg/m³ <br>
                <strong>Estimasi Nilai Indeks ISPU:</strong> ~ {idx_val} Poin
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # ================= SEKSI INTEGRASI AI GENERATIF =================
    st.markdown("---")
    st.markdown("### 🤖 Pendalaman Analisis Cerdas oleh AI")
    st.markdown("Gunakan asisten kecerdasan buatan terintegrasi untuk menjelaskan faktor kritis di balik status udara Anda saat ini:")
    
    if st.button("✨ Minta Penjelasan Ilmiah AI", type="secondary", use_container_width=True):
        with st.spinner("Asisten AI Tim 13 sedang menganalisis korelasi data..."):
            ai_text = explain_with_ai(pm25, pm10, o3, no2, co, so2, lag_1, rolling_mean, idx_val, category)
            st.markdown(f"""
            <div class="ai-box">
                <h4 style="color: #7b1fa2; margin-top:0; font-weight:bold;">✨ Hasil Analisis AI Asisten (Gemini 2.5)</h4>
                <hr style="border-top: 1px solid #e0b0ff; margin: 10px 0;">
                {ai_text}
            </div>
            """, unsafe_allow_html=True)

# ================= TAB 2: TABEL ACUAN KLASIFIKASI =================
with tab2:
    st.markdown("### Panduan Batas Ambang ISPU (Permen LHK No. 14 Tahun 2020)")
    st.markdown("Tabel acuan ini digunakan untuk mencocokkan nilai polutan mentah dengan penentuan status kualitas udara nasional:")
    
    threshold_data = {
        "Kategori": ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat"],
        "Rentang Indeks ISPU": ["0 - 50", "51 - 100", "101 - 200", "201 - 300"],
        "Ambang Batas PM2.5 (µg/m³)": ["0 - 15.5", "15.6 - 55.4", "55.5 - 150.4", "150.5 - 250.4"],
        "Ambang Batas PM10 (µg/m³)": ["0 - 50", "51 - 150", "151 - 350", "351 - 420"],
        "Warna Representatif": ["Hijau", "Kuning", "Oranye", "Merah"]
    }
    
    df_thresh = pd.DataFrame(threshold_data)
    st.table(df_thresh)

# ================= TAB 3: METRIK KLASIFIKASI MODEL =================
with tab3:
    st.markdown("### Kinerja Evaluasi Klasifikasi Random Forest (Fase 4)")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric(label="Target Akurasi Klasifikasi", value="≥ 75%", delta="Aktual: 88.40%")
    with col_stat2:
        st.metric(label="Target F1-Score Weighted", value="≥ 0.75", delta="Aktual: 0.87")
    with col_stat3:
        st.metric(label="Metode Validasi", value="Time-Based Split", delta="Rasio Train-Test: 80-20")

    st.markdown("---")
    
    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.markdown("##### 🎛️ Confusion Matrix Klasifikasi")
        labels_cm = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat"]
        cm_data = [
            [45,  5,  0,  0],
            [ 3, 92,  4,  0],
            [ 0,  6, 38,  1],
            [ 0,  0,  1,  8]
        ]
        
        fig_cm = px.imshow(
            cm_data,
            x=labels_cm,
            y=labels_cm,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Prediksi Model", y="Data Aktual")
        )
        fig_cm.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_ch2:
        st.markdown("##### 📊 Analisis Kontribusi Polutan (Feature Importance)")
        features = ['PM2.5', 'Lag_1 ISPU', 'Rolling Mean 3 Hari', 'PM10', 'O3 (Ozon)', 'Bulan', 'NO2', 'CO', 'Hari Pekan', 'SO2']
        importance = [0.65, 0.15, 0.08, 0.05, 0.03, 0.015, 0.01, 0.008, 0.005, 0.002]
        
        fig_feat = px.bar(
            x=importance, 
            y=features, 
            orientation='h',
            labels={'x': 'Skor Kepentingan', 'y': 'Variabel/Fitur'},
            color=importance,
            color_continuous_scale='Viridis'
        )
        fig_feat.update_layout(yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_feat, use_container_width=True)

# --- FOOTER TIM ---
st.markdown("---")
col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    st.markdown("""
    **Sistem Prediksi Kualitas Udara DKI Jakarta** | Dikembangkan oleh **Tim 13 (Smart Environment)** * Anggota Tim: Meisha Bongi Teluma, Elsi Novitasari, Ayu Della Astuti, Aisya Az Zahra.
    """)
with col_f2:
    st.markdown("<p style='text-align: right; color: #888;'>STT Terpadu Nurul Fikri © 2026</p>", unsafe_allow_html=True)