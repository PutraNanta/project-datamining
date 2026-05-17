import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def render_regression_tab():
    st.header("Algoritma: Regresi Linear (Prediksi)")
    
    # 1. Penjelasan Konsep
    st.subheader("1. Penjelasan Konsep")
    st.info("Regresi Linear adalah algoritma Supervised Learning yang digunakan untuk memprediksi nilai variabel yang bersifat numerik kontinu berdasarkan hubungan linear dengan variabel lain.")
    st.markdown("""
    Konsep utama dalam regresi:
    - **Variabel Target (Y)**: Nilai dependen yang ingin diprediksi (contoh: Harga Rumah, Total Penjualan, Nilai Ujian).
    - **Variabel Fitur (X)**: Nilai independen yang menjadi faktor penentu (contoh: Luas Tanah, Biaya Iklan, Jam Belajar).
    - Tujuan dari regresi linear adalah menarik sebuah **garis lurus terbaik (Best Fit Line)** yang meminimalkan jarak/error antara nilai aktual dengan prediksi model.
    """)
    
    # 2. Dataset
    st.subheader("2. Dataset")
    st.write("Silakan unggah dataset Anda sendiri atau gunakan dataset simulasi default.")
    
    uploaded_file = st.file_uploader("Unggah file CSV:", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset berhasil diunggah!")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return
    else:
        st.info("Menggunakan Dataset Default: Advertising (Simulasi)")
        # Membuat dummy data advertising
        np.random.seed(42)
        n_samples = 150
        tv_ads = np.random.uniform(10, 300, n_samples)
        radio_ads = np.random.uniform(0, 50, n_samples)
        newspaper_ads = np.random.uniform(0, 100, n_samples)
        # Sales memiliki hubungan kuat dengan TV dan Radio
        sales = 3.0 + 0.05 * tv_ads + 0.18 * radio_ads + np.random.normal(0, 1.5, n_samples)
        
        df = pd.DataFrame({
            "Biaya_TV": np.round(tv_ads, 1),
            "Biaya_Radio": np.round(radio_ads, 1),
            "Biaya_Koran": np.round(newspaper_ads, 1),
            "Total_Penjualan": np.round(sales, 1)
        })
        
    st.write("**Preview Data (10 Baris Pertama):**")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Total ukuran data: {df.shape[0]} Baris, {df.shape[1]} Kolom")
    
    # Pemilihan Target dan Fitur
    st.markdown("**Pengaturan Variabel**")
    col_names = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(col_names) < 2:
        st.error("Dataset harus memiliki minimal 2 kolom numerik untuk regresi.")
        return
        
    target_col = st.selectbox("Pilih Variabel Target / Label (Y):", col_names, index=len(col_names)-1)
    
    feature_options = [col for col in col_names if col != target_col]
    feature_cols = st.multiselect("Pilih Variabel Fitur (X):", feature_options, default=[feature_options[0]])
    
    # 3. Rumus
    st.subheader("3. Rumus")
    st.write("Persamaan matematis untuk Regresi Linear Berganda (Multiple Linear Regression):")
    st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon")
    st.write("Dimana:")
    st.markdown("""
    - $y$ = Variabel Target
    - $\\beta_0$ = Konstanta (Intercept)
    - $\\beta_i$ = Koefisien regresi (kemiringan garis) untuk fitur $x_i$
    - $x_i$ = Variabel Fitur ke-i
    - $\\epsilon$ = Error / Residu
    """)
    
    if not feature_cols:
        st.warning("Silakan pilih minimal 1 variabel fitur (X) untuk dapat melakukan perhitungan regresi.")
        return
        
    # 4. Step by step proses
    st.subheader("4. Step by step proses")
    st.write("Pemisahan Data (Train/Test Split):")
    test_size_percent = st.slider("Proporsi Data Testing (%)", min_value=10, max_value=50, value=20, step=5)
    test_size = test_size_percent / 100.0
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.markdown(f"""
    - **Total Data Pelatihan (Training)**: {len(X_train)} baris. Digunakan model untuk mempelajari pola.
    - **Total Data Pengujian (Testing)**: {len(X_test)} baris. Digunakan untuk mengevaluasi akurasi model pada data baru.
    """)
    
    # Proses Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    st.write("**Membangun Model Linear:**")
    eq_str = f"{target_col} = {model.intercept_:.3f}"
    for i, col in enumerate(feature_cols):
        eq_str += f" + ({model.coef_[i]:.3f} * {col})"
    st.code(eq_str, language="python")
    
    # 5. Hasil perhitungan/model
    st.subheader("5. Hasil Perhitungan (Evaluasi Model)")
    
    # Melakukan Prediksi
    y_pred = model.predict(X_test)
    
    # Menghitung Metrik
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric(label="MAE", value=f"{mae:.3f}")
    col_m2.metric(label="MSE", value=f"{mse:.3f}")
    col_m3.metric(label="RMSE", value=f"{rmse:.3f}")
    col_m4.metric(label="R² Score", value=f"{r2:.3f}")
    
    st.write("**Visualisasi Nilai Asli (Actual) vs Nilai Prediksi (Predicted):**")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, color='dodgerblue', alpha=0.7, edgecolors='black', label="Data Uji")
    
    # Garis diagonal sempurna
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prediksi Sempurna')
    
    ax.set_xlabel(f"Nilai Asli ({target_col})")
    ax.set_ylabel(f"Nilai Prediksi ({target_col})")
    ax.set_title("Grafik Actual vs Predicted")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    
    # 6. Interpretasi & 7. Kesimpulan
    st.subheader("6. Interpretasi & 7. Kesimpulan")
    
    st.write("**Interpretasi Metrik Evaluasi:**")
    st.markdown(f"""
    - **MAE ({mae:.3f})**: Secara rata-rata, tebakan prediksi kita meleset sebesar {mae:.3f} unit dari nilai aslinya.
    - **RMSE ({rmse:.3f})**: Akar rata-rata kuadrat error. Semakin kecil nilainya semakin baik. Berguna untuk mendeteksi error besar (karena dikuadratkan).
    - **R² Score ({r2:.3f})**: Menunjukkan bahwa model dapat menjelaskan **{r2*100:.1f}%** dari variabilitas nilai *{target_col}*. Sisanya ({100 - (r2*100):.1f}%) dijelaskan oleh faktor lain di luar model.
    """)
    
    # Kesimpulan berdasarkan R2 Score
    st.write("**Kesimpulan Kualitas Model:**")
    if r2 >= 0.8:
         st.success("Tingkat akurasi model **SANGAT BAIK**. Model berhasil menangkap tren data dengan kuat, sehingga sangat bisa diandalkan untuk prediksi data baru.")
    elif r2 >= 0.6:
         st.info("Tingkat akurasi model **CUKUP BAIK**. Model sudah bisa memprediksi tren secara lumayan, namun performanya masih bisa ditingkatkan (misalnya dengan menambah data atau fitur baru).")
    elif r2 >= 0.4:
         st.warning("Tingkat akurasi model **KURANG BAIK**. Tingkat error prediksi cukup tinggi. Hubungan linier antara fitur dan target kemungkinan lemah.")
    else:
         st.error("Tingkat akurasi model **BURUK**. Model tidak dapat menemukan hubungan linier yang jelas antara variabel. Kemungkinan besar datanya tidak cocok untuk regresi linear.")
