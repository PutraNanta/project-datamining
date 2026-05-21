import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def standardize_manual(features):
    values = features.to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    scaled_values = (values - means) / stds
    return scaled_values, means, stds


def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))


def kmeans_manual(values, k, max_iter=100):
    centroids = values[:k].copy()
    history = []
    labels = np.zeros(len(values), dtype=int)

    for iteration in range(1, max_iter + 1):
        distance_rows = []
        new_labels = []

        for idx, point in enumerate(values):
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            assigned_cluster = int(np.argmin(distances))
            new_labels.append(assigned_cluster)

            row = {"Data": idx + 1}
            for cluster_idx, distance in enumerate(distances):
                row[f"Jarak ke C{cluster_idx}"] = distance
            row["Cluster Terdekat"] = assigned_cluster
            distance_rows.append(row)

        new_labels = np.array(new_labels, dtype=int)
        new_centroids = centroids.copy()

        for cluster_idx in range(k):
            cluster_points = values[new_labels == cluster_idx]
            if len(cluster_points) > 0:
                new_centroids[cluster_idx] = cluster_points.mean(axis=0)

        history.append({
            "iteration": iteration,
            "centroids_before": centroids.copy(),
            "distances": pd.DataFrame(distance_rows),
            "labels": new_labels.copy(),
            "centroids_after": new_centroids.copy(),
        })

        if np.array_equal(labels, new_labels) and iteration > 1:
            labels = new_labels
            centroids = new_centroids
            break

        if np.allclose(centroids, new_centroids):
            labels = new_labels
            centroids = new_centroids
            break

        labels = new_labels
        centroids = new_centroids

    return labels, centroids, history


def inverse_standardize_manual(scaled_values, means, stds):
    return (scaled_values * stds) + means

def render_kmeans_tab():
    st.header("Algoritma: K-Means (Clustering)")
    
    # 1. Penjelasan Konsep
    st.subheader("1. Penjelasan Konsep")
    st.info("K-Means adalah algoritma Unsupervised Learning yang mengelompokkan data ke dalam beberapa cluster (kelompok) berdasarkan kemiripan (jarak terdekat).")
    st.markdown("""
    Konsep utama dalam K-Means:
    - **Clustering**: Proses pengelompokan data sehingga data dalam satu kelompok memiliki karakteristik yang mirip, sedangkan dengan data di kelompok lain berbeda.
    - **Centroid**: Titik pusat dari sebuah cluster. K-Means selalu berusaha mencari posisi centroid terbaik yang meminimalkan jarak titik-titik data ke centroidnya.
    - **Penentuan Centroid Awal**: Pada aplikasi ini centroid awal dipilih dari data pertama sampai data ke-K setelah standardisasi. Cara ini dipakai agar perhitungan manual bersifat tetap/deterministik dan mudah dicek ulang.
    - **Update Centroid**: Setelah data masuk ke cluster terdekat, centroid baru dihitung dari rata-rata setiap fitur milik seluruh data pada cluster tersebut.
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
    st.write("Standardisasi fitur:")
    st.latex(r"z = \frac{x - \bar{x}}{\sigma}")
    st.write("Jarak Euclidean antara titik $p$ dan centroid $c$:")
    st.latex(r"d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}")
    st.write("Update centroid:")
    st.latex(r"C_j = \frac{\sum x_i}{n_j}")
    st.write("Jika sebuah cluster memiliki beberapa anggota data, nilai centroid untuk setiap fitur adalah rata-rata nilai fitur tersebut pada anggota cluster.")
    st.write("Dimana $n$ adalah jumlah fitur/dimensi.")
    
    # 4. Step by step proses
    st.subheader("4. Step by step proses")
    
    st.markdown("**A. Pengaturan Parameter**")
    k = st.slider("Jumlah Cluster (K)", min_value=2, max_value=5, value=2, step=1)
    
    st.markdown("**B. Standardisasi Data Manual**")
    st.write("Karena variabel **Age** dan **Income** memiliki rentang nilai berbeda, setiap nilai dihitung manual menjadi z-score agar jarak Euclidean tidak didominasi oleh Income.")
    
    features = df[['Age', 'Income']]
    scaled_features, feature_means, feature_stds = standardize_manual(features)

    df_scale_params = pd.DataFrame({
        "Fitur": features.columns,
        "Mean": feature_means,
        "Standar Deviasi": feature_stds,
    })
    st.write("Mean dan standar deviasi yang dipakai:")
    st.dataframe(df_scale_params, use_container_width=True)
    
    df_scaled = pd.DataFrame(scaled_features, columns=['Age (Scaled)', 'Income (Scaled)'])
    st.write("Hasil standardisasi:")
    st.dataframe(df_scaled, use_container_width=True)
    
    st.markdown("**C. Inisialisasi Centroid Manual**")
    st.write(f"Centroid awal diambil dari {k} data pertama agar perhitungan deterministik dan mudah diikuti.")
    st.markdown(f"""
    Karena nilai **K = {k}**, maka centroid awal yang digunakan adalah:
    - **C0** = data pelanggan ke-1 setelah standardisasi.
    - **C1** = data pelanggan ke-2 setelah standardisasi.
    {f"- **C2** = data pelanggan ke-3 setelah standardisasi." if k >= 3 else ""}
    {f"- **C3** = data pelanggan ke-4 setelah standardisasi." if k >= 4 else ""}
    {f"- **C4** = data pelanggan ke-5 setelah standardisasi." if k >= 5 else ""}

    Setelah semua data dihitung jaraknya ke centroid, setiap data dimasukkan ke centroid dengan jarak paling kecil.
    Centroid berikutnya tidak dipilih lagi dari data awal, tetapi dihitung menggunakan rata-rata anggota cluster:
    """)
    st.latex(r"C_{cluster, fitur} = \frac{x_1 + x_2 + ... + x_n}{n}")
    
    labels, centroids_scaled, iteration_history = kmeans_manual(scaled_features, k)
    df_initial_centroids = pd.DataFrame(
        iteration_history[0]["centroids_before"],
        columns=['Age (Scaled)', 'Income (Scaled)']
    )
    df_initial_centroids.index.name = 'Centroid'
    st.dataframe(df_initial_centroids, use_container_width=True)

    st.markdown("**D. Iterasi K-Means Manual**")
    st.write("Pada setiap iterasi, jarak setiap data ke semua centroid dihitung dengan rumus Euclidean. Data masuk ke cluster dengan jarak paling kecil, lalu centroid baru dihitung dari rata-rata data di cluster tersebut.")

    for step in iteration_history:
        with st.expander(f"Iterasi {step['iteration']}", expanded=step["iteration"] == 1):
            st.write("Jarak setiap data ke centroid:")
            st.dataframe(step["distances"], use_container_width=True)

            df_centroid_update = pd.DataFrame(
                step["centroids_after"],
                columns=['Age (Scaled)', 'Income (Scaled)']
            )
            df_centroid_update.index.name = 'Centroid Baru'
            st.write("Centroid baru dari rata-rata anggota cluster:")
            st.dataframe(df_centroid_update, use_container_width=True)
    
    # Menyimpan label ke dataframe asli
    df['Cluster'] = labels
    
    # 5. Hasil perhitungan/model
    st.subheader("5. Hasil Perhitungan & Visualisasi")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Hasil Klasterisasi per Pelanggan:**")
        st.dataframe(df, use_container_width=True)
        
    with col2:
        st.write("**Posisi Centroid (Skala Asli):**")
        # Mengembalikan titik centroid dari skala standar ke skala asli agar mudah diinterpretasikan.
        centroids_original = inverse_standardize_manual(centroids_scaled, feature_means, feature_stds)
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
