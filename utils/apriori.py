import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def render_apriori_tab():
    st.header("Algoritma: Apriori (Association Rule)")
    
    # 1. Penjelasan Konsep
    st.subheader("1. Penjelasan Konsep")
    st.info("Algoritma Apriori adalah metode untuk mencari pola hubungan antar item dalam suatu dataset transaksi (Market Basket Analysis).")
    st.markdown("""
    Konsep utama dari algoritma ini didasarkan pada 3 metrik dasar:
    - **Support**: Menunjukkan seberapa sering suatu item atau kombinasi item muncul dalam keseluruhan transaksi.
    - **Confidence**: Menunjukkan seberapa kuat hubungan antar item. Berapa probabilitas item B dibeli jika item A sudah dibeli.
    - **Lift**: Menunjukkan rasio peluang item A dan B dibeli bersamaan dibandingkan jika keduanya dibeli secara acak independen. Jika Lift > 1, maka terdapat hubungan saling terkait yang positif.
    """)
    
    # 2. Dataset
    st.subheader("2. Dataset")
    st.write("Menggunakan dataset riwayat transaksi belanja (Default):")
    
    dataset = [
        ["Susu", "Gula", "Teh"],
        ["Teh", "Gula", "Roti"],
        ["Teh", "Gula"],
        ["Susu", "Roti"],
        ["Susu", "Gula", "Roti"],
        ["Teh", "Gula"],
        ["Gula", "Kopi", "Susu"],
        ["Gula", "Kopi", "Susu"],
        ["Susu", "Roti", "Kopi"],
        ["Gula", "Teh", "Kopi"]
    ]
    
    df_transaksi = pd.DataFrame({
        "ID Transaksi": [f"Transaksi {i+1}" for i in range(len(dataset))],
        "Daftar Item": [", ".join(items) for items in dataset]
    })
    st.table(df_transaksi)
    
    # 3. Rumus
    st.subheader("3. Rumus")
    st.latex(r"Support(A) = \frac{\text{Jumlah Transaksi Mengandung } A}{\text{Total Transaksi}}")
    st.latex(r"Support(A, B) = \frac{\text{Jumlah Transaksi Mengandung } A \text{ dan } B}{\text{Total Transaksi}}")
    st.latex(r"Confidence(A \rightarrow B) = \frac{Support(A, B)}{Support(A)}")
    st.latex(r"Lift(A \rightarrow B) = \frac{Confidence(A \rightarrow B)}{Support(B)}")
    
    # 4. Step by step proses
    st.subheader("4. Step by step proses")
    st.write("Pengaturan Parameter:")
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Minimum Support", min_value=0.1, max_value=1.0, value=0.2, step=0.1)
    with col2:
        min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        
    # Preprocessing
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Apriori
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=3)
    
    if frequent_itemsets.empty:
        st.warning("Tidak ada itemset yang memenuhi nilai Minimum Support.")
    else:
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        
        st.markdown("**A. Frequent 1-Itemset**")
        itemset_1 = frequent_itemsets[frequent_itemsets['length'] == 1].copy()
        if not itemset_1.empty:
            itemset_1['itemsets'] = itemset_1['itemsets'].apply(lambda x: ", ".join(list(x)))
            st.dataframe(itemset_1[['itemsets', 'support']], use_container_width=True)
        else:
            st.write("Tidak ada.")
            
        st.markdown("**B. Frequent 2-Itemset**")
        itemset_2 = frequent_itemsets[frequent_itemsets['length'] == 2].copy()
        if not itemset_2.empty:
            itemset_2['itemsets'] = itemset_2['itemsets'].apply(lambda x: ", ".join(list(x)))
            st.dataframe(itemset_2[['itemsets', 'support']], use_container_width=True)
        else:
            st.write("Tidak ada.")
            
        st.markdown("**C. Frequent 3-Itemset**")
        itemset_3 = frequent_itemsets[frequent_itemsets['length'] == 3].copy()
        if not itemset_3.empty:
            itemset_3['itemsets'] = itemset_3['itemsets'].apply(lambda x: ", ".join(list(x)))
            st.dataframe(itemset_3[['itemsets', 'support']], use_container_width=True)
        else:
            st.write("Tidak ada.")

    # 5. Hasil perhitungan/model
    st.subheader("5. Hasil Perhitungan (Association Rules)")
    if frequent_itemsets.empty:
         st.warning("Tidak dapat menghasilkan rules karena frequent itemset kosong.")
    else:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        if rules.empty:
            st.warning("Tidak ada association rules yang memenuhi nilai Minimum Confidence.")
        else:
            rules_show = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
            rules_show['antecedents'] = rules_show['antecedents'].apply(lambda x: ", ".join(list(x)))
            rules_show['consequents'] = rules_show['consequents'].apply(lambda x: ", ".join(list(x)))
            
            st.dataframe(rules_show, use_container_width=True)
            
            # 6. Interpretasi & 7. Kesimpulan
            st.subheader("6. Interpretasi & 7. Kesimpulan")
            
            # Mendapatkan rule terbaik berdasarkan lift tertinggi
            best_rule = rules.loc[rules['lift'].idxmax()]
            ant = ", ".join(list(best_rule['antecedents']))
            con = ", ".join(list(best_rule['consequents']))
            conf_percent = best_rule['confidence'] * 100
            supp_percent = best_rule['support'] * 100
            lift_val = best_rule['lift']
            
            st.success(f"**Rule Terbaik:** Jika pelanggan membeli **{ant}**, maka mereka kemungkinan besar juga akan membeli **{con}**.")
            
            st.markdown(f"""
            **Interpretasi Metrik:**
            - **Support ({supp_percent:.0f}%)**: Sebanyak {supp_percent:.0f}% dari total transaksi di dataset mengandung kombinasi {ant} dan {con}.
            - **Confidence ({conf_percent:.0f}%)**: Dari seluruh pelanggan yang membeli {ant}, sebanyak {conf_percent:.0f}% pelanggan di antaranya pasti juga membeli {con}.
            - **Lift ({lift_val:.2f})**: Nilai Lift sebesar {lift_val:.2f} (lebih dari 1) menunjukkan bahwa {ant} dan {con} memiliki keterkaitan yang sangat kuat. Membeli {ant} meningkatkan probabilitas seseorang untuk membeli {con} sebesar {lift_val:.2f} kali lipat.
            
            **Kesimpulan / Rekomendasi Bisnis:**
            Berdasarkan rule terbaik di atas, produk **{ant}** dan **{con}** sangat disarankan untuk:
            1. Diletakkan di rak toko yang saling berdekatan.
            2. Dijadikan satu paket promosi bundling (contoh: "Gratis {con} setiap pembelian {ant}").
            """)
