import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

def render_c45_tab():
    st.header("Algoritma: Klasifikasi C4.5 (Decision Tree)")
    
    # 1. Penjelasan Konsep
    st.subheader("1. Penjelasan Konsep")
    st.info("Algoritma C4.5 adalah metode pemodelan klasifikasi yang menghasilkan sebuah **Pohon Keputusan (Decision Tree)**. Algoritma ini memecah sekumpulan data secara bertahap berdasarkan fitur yang memberikan informasi paling signifikan.")
    st.markdown("""
    **Metrik Utama dalam Pembentukan Pohon (C4.5):**
    - **Entropy**: Ukuran ketidakteraturan atau "kebingungan" dalam suatu himpunan data. Semakin murni (satu kelas dominan) suatu himpunan, semakin mendekati 0 entropinya.
    - **Information Gain**: Pengurangan nilai entropi setelah dataset dipecah berdasarkan suatu atribut. Atribut dengan Gain tertinggi akan dipilih sebagai cabang pohon.
    - **Gain Ratio**: Pengembangan dari Information Gain yang meminimalkan bias (kecenderungan salah pilih) terhadap atribut yang memiliki banyak nilai unik.
    """)
    
    # 2. Dataset
    st.subheader("2. Dataset")
    st.write("Unggah dataset Anda sendiri atau gunakan dataset prediksi Kelulusan Mahasiswa.")
    
    uploaded_file = st.file_uploader("Unggah Dataset CSV (Klasifikasi):", type=["csv"], key="c45_uploader")
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success("Dataset berhasil diunggah!")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return
    else:
        st.info("Menggunakan Dataset Default (Prediksi Kelulusan Mahasiswa):")
        # Membuat dataset dummy yang representatif
        np.random.seed(42)
        n = 200
        ipk = np.random.uniform(2.0, 4.0, n)
        kehadiran = np.random.randint(50, 100, n)
        tugas = np.random.choice(["Lengkap", "Tidak Lengkap"], n, p=[0.75, 0.25])
        organisasi = np.random.choice(["Aktif", "Pasif"], n)
        
        # Logika penentuan label secara deterministik parsial agar model bisa belajar
        status = []
        for i in range(n):
            if ipk[i] >= 3.2 and kehadiran[i] >= 75 and tugas[i] == "Lengkap":
                status.append("Tepat Waktu")
            elif ipk[i] >= 3.6:
                status.append("Tepat Waktu")
            elif ipk[i] < 2.5:
                status.append("Terlambat")
            elif kehadiran[i] < 70 or tugas[i] == "Tidak Lengkap":
                status.append("Terlambat")
            else:
                # Random noise
                status.append(np.random.choice(["Tepat Waktu", "Terlambat"]))
                
        df_raw = pd.DataFrame({
            "IPK": np.round(ipk, 2),
            "Persentase_Kehadiran": kehadiran,
            "Status_Tugas": tugas,
            "Ikut_Organisasi": organisasi,
            "Kelulusan": status
        })
        
    st.write("**Preview Data:**")
    st.dataframe(df_raw.head(10), use_container_width=True)
    
    # Pemilihan target
    col_names = df_raw.columns.tolist()
    target_col = st.selectbox(
        "Pilih Target Class (Label yang ingin diklasifikasikan):",
        col_names,
        index=len(col_names)-1,
        key="c45_target_col",
    )
    
    # Preprocessing (Encoding)
    df_processed = df_raw.copy()
    
    st.write("**Encoding Data Kategorikal**")
    st.write("Model *Decision Tree* pada Python (scikit-learn) membutuhkan input numerik, sehingga kolom berformat teks (kategorikal) harus diubah menjadi angka/numerik.")
    
    # Encode Target Label
    le_target = LabelEncoder()
    df_processed[target_col] = le_target.fit_transform(df_processed[target_col].astype(str))
    target_names = [str(cls) for cls in le_target.classes_]
    
    # Tampilkan mapping label target
    mapping_dict = dict(zip(le_target.transform(le_target.classes_), le_target.classes_))
    st.caption(f"Mapping Target: {mapping_dict}")
    
    # Pisahkan X dan Y
    feature_cols = [c for c in col_names if c != target_col]
    X = df_processed[feature_cols]
    
    # One-hot encode fitur kategorikal (dummy variables)
    X_encoded = pd.get_dummies(X, drop_first=True)
    y = df_processed[target_col]
    
    st.dataframe(X_encoded.head(5), use_container_width=True)
    
    # 3. Rumus
    st.subheader("3. Rumus")
    st.latex(r"Entropy(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)")
    st.latex(r"Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \left( \frac{|S_v|}{|S|} \right) Entropy(S_v)")
    st.latex(r"SplitInfo(S, A) = - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2 \left(\frac{|S_v|}{|S|}\right)")
    st.latex(r"GainRatio = \frac{Gain(S, A)}{SplitInfo(S, A)}")
    
    # 4. Step by step proses
    st.subheader("4. Step by step proses")
    st.write("Pengaturan Data & Model:")
    
    col_a, col_b = st.columns(2)
    with col_a:
        test_size_percent = st.slider(
            "Proporsi Data Testing (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            key="c45_test_size_percent",
        )
    with col_b:
        max_depth = st.slider(
            "Maksimal Kedalaman Pohon (Max Depth)",
            min_value=1,
            max_value=10,
            value=4,
            help="Mencegah Overfitting",
            key="c45_max_depth",
        )
        
    test_size = test_size_percent / 100.0
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=42)
    st.write(f"- Data Training: {len(X_train)} baris")
    st.write(f"- Data Testing: {len(X_test)} baris")
    
    # Model Training menggunakan Entropy untuk mereplikasi C4.5
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Hasil perhitungan/model
    st.subheader("5. Hasil Perhitungan & Evaluasi Model")
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.metric(label="Akurasi Model", value=f"{acc * 100:.2f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm, xticklabels=target_names, yticklabels=target_names)
        ax_cm.set_xlabel('Hasil Prediksi (Predicted)')
        ax_cm.set_ylabel('Hasil Sebenarnya (Actual)')
        st.pyplot(fig_cm)
        
    with col2:
        st.write("**Classification Report:**")
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format("{:.2f}"))
        
    st.write("**Visualisasi Pohon Keputusan (Decision Tree):**")
    fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
    # Render the tree plot
    plot_tree(model, feature_names=X_encoded.columns.tolist(), class_names=target_names, filled=True, rounded=True, fontsize=10, ax=ax_tree)
    st.pyplot(fig_tree)
    
    # 6. Interpretasi & 7. Kesimpulan
    st.subheader("6. Interpretasi Aturan & 7. Kesimpulan")
    
    # Mengekspor text rules
    tree_rules = export_text(model, feature_names=X_encoded.columns.tolist())
    
    st.write("**Aturan Keputusan (Decision Rules):**")
    st.code(tree_rules, language="text")
    
    st.success(f"""
    **Kesimpulan Evaluasi:**
    - Model *Decision Tree* (C4.5) berhasil dilatih dan dievaluasi dengan tingkat akurasi sebesar **{acc * 100:.2f}%**.
    - Berdasarkan struktur pohon keputusan di atas, kita dapat mengekstrak aturan logika `IF-THEN` yang sangat intuitif untuk dipahami oleh manusia (contoh: *Jika IPK > 3.2 dan Kehadiran > 75, maka Tepat Waktu*).
    - Keuntungan utama algoritma C4.5 adalah sifat pemodelannya yang *white-box*, artinya alur pemikirannya dapat divisualisasikan dengan jelas dan mudah dikonversi menjadi baris kode *if-else* di dalam sistem aplikasi nyata.
    """)
