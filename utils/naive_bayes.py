import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

def render_naive_bayes_tab():
    st.header("Algoritma: Naive Bayes (Probabilistik)")
    
    # 1. Penjelasan Konsep
    st.subheader("1. Penjelasan Konsep")
    st.info("Naive Bayes adalah algoritma klasifikasi berbasis statistik yang menggunakan **Teorema Bayes**. Algoritma ini memprediksi peluang (probabilitas) suatu data masuk ke dalam kelas tertentu.")
    st.markdown("""
    Konsep probabilitas utama dalam Naive Bayes:
    - **Prior Probability $P(C)$**: Peluang awal atau probabilitas historis munculnya sebuah kelas (contoh: Seberapa sering seseorang terkena flu di musim hujan).
    - **Likelihood $P(X|C)$**: Peluang kemunculan nilai fitur $X$ jika diketahui kelasnya adalah $C$.
    - **Posterior Probability $P(C|X)$**: Peluang akhir (hasil prediksi). Yakni peluang bahwa kelasnya adalah $C$ setelah kita mengamati bukti/fitur $X$.
    - **Asumsi "Naive" (Naif)**: Algoritma ini berasumsi bahwa setiap variabel/fitur bekerja secara *independen* (tidak saling mempengaruhi) satu sama lain.
    """)
    
    # 2. Dataset
    st.subheader("2. Dataset")
    st.write("Silakan unggah dataset klasifikasi atau gunakan dataset default Prediksi Risiko Penyakit.")
    
    uploaded_file = st.file_uploader("Unggah Dataset CSV (Naive Bayes):", type=["csv"], key="nb_uploader")
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success("Dataset berhasil diunggah!")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return
    else:
        st.info("Menggunakan Dataset Default (Prediksi Risiko Penyakit Jantung):")
        np.random.seed(42)
        n = 200
        usia = np.random.normal(55, 10, n)
        tekanan_darah = np.random.normal(120, 15, n)
        kolesterol = np.random.normal(200, 30, n)
        
        # Logika probabilitas (hidden logic) untuk mengenerate label berdasar fitur
        z = (usia - 55)/10 + (tekanan_darah - 120)/15 + (kolesterol - 200)/30
        prob = 1 / (1 + np.exp(-z)) # sigmoid function
        
        status = ["Beresiko" if p > 0.5 else "Sehat" for p in prob]
        gender = np.random.choice(["Pria", "Wanita"], n)
        
        df_raw = pd.DataFrame({
            "Usia": np.round(usia, 0),
            "Tekanan_Darah": np.round(tekanan_darah, 0),
            "Kolesterol": np.round(kolesterol, 0),
            "Jenis_Kelamin": gender,
            "Diagnosa": status
        })
        
    st.write("**Preview Data:**")
    st.dataframe(df_raw.head(10), use_container_width=True)
    
    # Pemilihan target
    col_names = df_raw.columns.tolist()
    target_col = st.selectbox("Pilih Target Class (Y):", col_names, index=len(col_names)-1, key="nb_target")
    
    # Preprocessing & Encoding
    st.markdown("**Preprocessing & Encoding Data Kategorikal**")
    df_processed = df_raw.copy()
    
    # Encode target
    le_target = LabelEncoder()
    df_processed[target_col] = le_target.fit_transform(df_processed[target_col].astype(str))
    target_names = [str(cls) for cls in le_target.classes_]
    
    st.caption(f"Mapping Class Target: {dict(zip(le_target.transform(le_target.classes_), le_target.classes_))}")
    
    feature_cols = [c for c in col_names if c != target_col]
    X = df_processed[feature_cols]
    
    # One-hot encoding untuk fitur x (jika ada string kategorikal)
    X_encoded = pd.get_dummies(X, drop_first=True)
    y = df_processed[target_col]
    
    st.dataframe(X_encoded.head(5), use_container_width=True)
    
    # 3. Rumus
    st.subheader("3. Rumus (Teorema Bayes)")
    st.latex(r"P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}")
    st.write("Penjelasan Variabel:")
    st.markdown("""
    - **$P(C|X)$**: *Posterior* - Peluang akhir diprediksinya kelas C dengan memperhitungkan kondisi atribut X.
    - **$P(X|C)$**: *Likelihood* - Peluang atribut X terjadi secara historis apabila telah diketahui kelasnya adalah C.
    - **$P(C)$**: *Prior* - Peluang dasar munculnya kelas C dari keseluruhan data.
    - **$P(X)$**: *Evidence* - Peluang dasar munculnya atribut X.
    """)
    
    # 4. Step by step proses
    st.subheader("4. Step by step proses")
    test_size_percent = st.slider("Proporsi Data Testing (%) ", min_value=10, max_value=50, value=20, step=5, key="nb_slider")
    test_size = test_size_percent / 100.0
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=42)
    st.write(f"- Melatih **Gaussian Naive Bayes** menggunakan {len(X_train)} data latih (Train).")
    st.write(f"- Menguji model menggunakan {len(X_test)} data uji (Test).")
    
    # Model Training
    # GaussianNB ideal karena dataset dummy di atas menggunakan distribusi kurva normal / numerik kontinu
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # 5. Hasil perhitungan/model
    st.subheader("5. Hasil Perhitungan & Evaluasi Model")
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.metric(label="Akurasi Model GaussianNB", value=f"{acc * 100:.2f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax, xticklabels=target_names, yticklabels=target_names)
        ax.set_xlabel('Hasil Prediksi (Predicted)')
        ax.set_ylabel('Data Aktual (Actual)')
        st.pyplot(fig)
        
    with col2:
        st.write("**Classification Report:**")
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format("{:.2f}"))
        
    st.write("**Preview Hasil Prediksi Model pada Data Uji:**")
    df_pred = X_test.copy().head(10)
    df_pred['Aktual (Actual)'] = le_target.inverse_transform(y_test[:10])
    df_pred['Prediksi Model'] = le_target.inverse_transform(y_pred[:10])
    
    # Fungsi pewarnaan: Hijau jika benar, Merah jika salah
    def color_result(row):
        if row['Aktual (Actual)'] == row['Prediksi Model']:
            return ['background-color: #d4edda; color: black'] * len(row)
        else:
            return ['background-color: #f8d7da; color: black'] * len(row)
            
    st.dataframe(df_pred.style.apply(color_result, axis=1), use_container_width=True)
    
    # 6. Interpretasi & 7. Kesimpulan
    st.subheader("6. Interpretasi & 7. Kesimpulan")
    
    st.success(f"""
    **Interpretasi Model:**
    Model Gaussian Naive Bayes berhasil mengklasifikasikan data dengan tingkat akurasi mencapai **{acc * 100:.2f}%**. 
    Pendekatan **Gaussian** (*GaussianNB*) digunakan dalam implementasi ini karena sangat efisien dalam menangani fitur berbentuk numerik kontinu (seperti Usia, Kolesterol, Tekanan Darah). Algoritma ini secara matematis menghitung probabilitas data berdasarkan rentang nilai rata-rata (*mean*) dan standar deviasinya menyerupai kurva lonceng normal.
    
    **Kesimpulan Klasifikasi:**
    - Meskipun teori di balik Naive Bayes mengasumsikan bahwa fitur beroperasi secara absolut independen (suatu kondisi "Naif" yang jarang terjadi di dunia nyata), performa model ini tetap kompetitif.
    - Bukti pengujian (tabel prediksi di atas) menunjukkan bahwa Probabilitas Posterior $P(C|X)$ mampu mengeksekusi perhitungan klasifikasi yang akurat dengan kecepatan pemrosesan (komputasi) yang jauh lebih cepat dibandingkan algoritma pemodelan lainnya. Algoritma ini sangat ideal untuk studi kasus yang memiliki dataset dalam skala raksasa.
    """)
