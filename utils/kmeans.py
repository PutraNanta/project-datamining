import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def render_kmeans_tab():
    st.header("Algoritma: K-Means (Clustering)")
    
    # 1. Penjelasan Konsep
    st.subheader("1. Penjelasan Konsep")
    st.info("K-Means adalah algoritma Unsupervised Learning yang mengelompokkan data ke dalam beberapa cluster (kelompok) berdasarkan kemiripan (jarak terdekat).")
    st.markdown("""
    Konsep utama dalam K-Means:
    - **Clustering**: Proses pengelompokan data sehingga data dalam satu kelompok memiliki karakteristik yang mirip, sedangkan dengan data di kelompok lain berbeda.
    - **Centroid**: Titik pusat dari sebuah cluster. K-Means selalu berusaha mencari posisi centroid terbaik yang meminimalkan jarak titik-titik data ke centroidnya.
    - **Jarak Euclidean (Euclidean Distance)**: Metrik pengukuran jarak garis lurus terpendek antara dua titik data. K-Means menggunakan jarak ini untuk menentukan cluster mana yang paling dekat dengan suatu titik data.
    """)
    
    # 2. Dataset
    st.subheader("2. Dataset")
    st.write("Menggunakan dataset pelanggan (Default):")
    
    data = {
        "CustID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Age": [41, 47, 33, 29, 47, 40, 38, 42, 26, 47],
        "Income": [19, 100, 57, 19, 253, 81, 56, 64, 18, 115]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
    # 3. Rumus
    st.subheader("3. Rumus")
    st.write("Jarak Euclidean antara titik $p$ dan $q$:")
    st.latex(r"d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}")
    st.write("Dimana $n$ adalah jumlah fitur/dimensi.")
    
    # 4. Step by step proses
    st.subheader("4. Step by step proses")
    
    st.markdown("**A. Pengaturan Parameter**")
    k = st.slider("Jumlah Cluster (K)", min_value=2, max_value=5, value=2, step=1)
    
    st.markdown("**B. Scaling Data (Standardization)**")
    st.write("Karena variabel **Age** dan **Income** memiliki rentang nilai yang berbeda (puluhan vs ratusan), kita perlu menstandarkan datanya. Jika tidak, perhitungan jarak Euclidean akan didominasi oleh variabel dengan nilai yang lebih besar (Income).")
    
    features = df[['Age', 'Income']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    df_scaled = pd.DataFrame(scaled_features, columns=['Age (Scaled)', 'Income (Scaled)'])
    st.dataframe(df_scaled.head(), use_container_width=True)
    
    st.markdown("**C. Proses K-Means**")
    st.write(f"Algoritma menginisialisasi {k} titik centroid secara acak, kemudian menghitung jarak semua data ke centroid tersebut, lalu mengelompokkan data ke centroid terdekat. Posisi centroid akan terus diperbarui (digeser ke rata-rata titik di clusternya) hingga posisinya konvergen (tidak berubah lagi).")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    
    # Menyimpan label ke dataframe asli
    df['Cluster'] = kmeans.labels_
    
    # 5. Hasil perhitungan/model
    st.subheader("5. Hasil Perhitungan & Visualisasi")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Hasil Klasterisasi per Pelanggan:**")
        st.dataframe(df, use_container_width=True)
        
    with col2:
        st.write("**Posisi Centroid (Skala Asli):**")
        # Mengembalikan titik centroid dari skala standard ke skala asli agar mudah diinterpretasikan
        centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
        df_centroids = pd.DataFrame(centroids_original, columns=['Age', 'Income'])
        df_centroids.index.name = 'Cluster'
        st.dataframe(df_centroids, use_container_width=True)
        
    st.write("**Visualisasi Scatter Plot (Age vs Income):**")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Membuat palet warna yang dinamis sesuai jumlah K
    palette = sns.color_palette("Set1", n_colors=k)
    
    sns.scatterplot(data=df, x='Age', y='Income', hue='Cluster', palette=palette, s=150, ax=ax)
    
    # Plot centroid
    ax.scatter(centroids_original[:, 0], centroids_original[:, 1], color='black', marker='X', s=300, label='Centroid', zorder=10)
    ax.set_title(f"Segmentasi {k} Cluster", fontsize=14)
    ax.legend(title='Cluster')
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    
    # 6. Interpretasi & 7. Kesimpulan
    st.subheader("6. Interpretasi & 7. Kesimpulan")
    
    st.write("**Karakteristik Masing-masing Cluster:**")
    for i in range(k):
        cluster_data = df[df['Cluster'] == i]
        avg_age = cluster_data['Age'].mean()
        avg_income = cluster_data['Income'].mean()
        count = len(cluster_data)
        
        # Logika heuristik sederhana untuk membuat deskripsi teks otomatis
        age_desc = "muda" if avg_age < 35 else ("menengah" if avg_age <= 45 else "dewasa/tua")
        income_desc = "rendah" if avg_income < 40 else ("menengah" if avg_income <= 85 else "tinggi")
        
        st.markdown(f"- **Cluster {i}** ({count} pelanggan): Rata-rata usia **{avg_age:.1f} tahun** (kategori usia {age_desc}) dengan rata-rata pendapatan **{avg_income:.1f}** (pendapatan {income_desc}).")
        
    st.success("""
    **Kesimpulan Segmentasi Pelanggan:**\n
    Berdasarkan karakteristik klasterisasi K-Means di atas, perusahaan dapat menerapkan strategi pemasaran yang lebih personal (Targeted Marketing):
    - Pelanggan di klaster berpendapatan tinggi adalah **VIP Customer**. Tim marketing bisa menawarkan produk-produk premium (upselling/cross-selling) dan layanan prioritas.
    - Pelanggan di klaster usia muda dengan pendapatan rendah/menengah dapat diberikan strategi pemasaran berbasis diskon, promo cashback, atau produk entry-level yang terjangkau.
    - Pelanggan di kategori usia dan pendapatan menengah merupakan pengguna potensial yang perlu dipertahankan dengan loyalty program (program poin, member khusus).
    """)
