import streamlit as st

# Konfigurasi Halaman
st.set_page_config(
    page_title="Data Mining Learning App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Aplikasi
st.title("📊 Aplikasi Pembelajaran Data Mining")
st.markdown("---")

# Sidebar Penjelasan Singkat
with st.sidebar:
    st.header("💡 Tentang Aplikasi")
    st.write(
        "Aplikasi ini merupakan platform interaktif untuk mempelajari "
        "berbagai algoritma utama dalam **Data Mining**. "
        "Setiap algoritma dijelaskan secara komprehensif mulai dari konsep, "
        "rumus, hingga contoh perhitungannya."
    )
    st.write("Dibuat untuk keperluan edukasi dan eksplorasi data.")
    st.markdown("---")
    st.caption("© 2026 - Data Mining Project")

# Fungsi untuk merender template struktur konten setiap tab
def render_algorithm_template(algo_name):
    st.header(f"Algoritma: {algo_name}")
    
    st.subheader("1. Penjelasan Konsep")
    st.info(f"Bagian ini akan berisi penjelasan teori dan konsep dasar mengenai algoritma {algo_name}.")
    
    st.subheader("2. Dataset")
    st.write("Area ini akan digunakan untuk mengunggah dataset (.csv) atau memilih dataset bawaan untuk demonstrasi.")
    
    st.subheader("3. Step by step proses")
    st.write("Bagian ini akan menjabarkan alur kerja algoritma secara berurutan, dari awal hingga akhir.")
    
    st.subheader("4. Rumus")
    st.write("Rumus-rumus matematika atau statistik yang mendasari perhitungan algoritma akan ditampilkan dengan format LaTeX.")
    
    st.subheader("5. Hasil perhitungan/model")
    st.write("Output visualisasi, metrik, rules, atau model yang dihasilkan setelah algoritma dijalankan pada dataset.")
    
    st.subheader("6. Interpretasi")
    st.write("Penjelasan praktis mengenai cara membaca, memahami, dan memanfaatkan hasil yang telah didapat.")
    
    st.subheader("7. Kesimpulan")
    st.success("Ringkasan mengenai apa yang dapat disimpulkan dari penerapan algoritma ini pada data terkait.")

# Membuat 5 tab utama dalam satu aplikasi
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛒 1. Apriori", 
    "🎯 2. K-Means", 
    "📈 3. Regression", 
    "🌳 4. Klasifikasi C4.5", 
    "📊 5. Naive Bayes"
])

# Mengisi konten masing-masing tab
with tab1:
    render_algorithm_template("Apriori (Association Rule)")

with tab2:
    render_algorithm_template("K-Means (Clustering)")

with tab3:
    render_algorithm_template("Regression (Prediksi)")

with tab4:
    render_algorithm_template("Klasifikasi C4.5 (Decision Tree)")

with tab5:
    render_algorithm_template("Naive Bayes (Probabilistik)")
