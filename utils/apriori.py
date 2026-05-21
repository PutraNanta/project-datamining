import streamlit as st
import pandas as pd
from itertools import combinations


def format_itemset(itemset):
    return ", ".join(sorted(itemset))


def encode_transactions_manual(dataset):
    items = sorted({item for transaction in dataset for item in transaction})
    rows = []

    for idx, transaction in enumerate(dataset, start=1):
        transaction_set = set(transaction)
        row = {"ID Transaksi": f"Transaksi {idx}"}
        for item in items:
            row[item] = 1 if item in transaction_set else 0
        rows.append(row)

    return pd.DataFrame(rows), items


def calculate_support_count(dataset, candidate):
    candidate_set = set(candidate)
    return sum(1 for transaction in dataset if candidate_set.issubset(set(transaction)))


def generate_candidates(items, size, previous_frequent):
    candidates = []
    previous_sets = set(previous_frequent)

    for candidate in combinations(items, size):
        candidate_set = frozenset(candidate)
        if size == 1:
            candidates.append(candidate_set)
            continue

        subsets_are_frequent = all(
            frozenset(subset) in previous_sets
            for subset in combinations(candidate_set, size - 1)
        )
        if subsets_are_frequent:
            candidates.append(candidate_set)

    return candidates


def apriori_manual(dataset, min_support, max_len=3):
    items = sorted({item for transaction in dataset for item in transaction})
    total_transactions = len(dataset)
    support_map = {}
    steps = []
    previous_frequent = []

    for size in range(1, max_len + 1):
        candidates = generate_candidates(items, size, previous_frequent)
        candidate_rows = []
        current_frequent = []

        for candidate in candidates:
            support_count = calculate_support_count(dataset, candidate)
            support = support_count / total_transactions
            support_map[candidate] = support

            row = {
                "Itemset": format_itemset(candidate),
                "Jumlah Transaksi": support_count,
                "Support": support,
                "Status": "Lolos" if support >= min_support else "Tidak Lolos",
            }
            candidate_rows.append(row)

            if support >= min_support:
                current_frequent.append(candidate)

        steps.append({
            "size": size,
            "candidates": pd.DataFrame(candidate_rows),
            "frequent": current_frequent,
        })

        if not current_frequent:
            break

        previous_frequent = current_frequent

    frequent_itemsets = []
    for itemset, support in support_map.items():
        if support >= min_support:
            frequent_itemsets.append({
                "itemsets": itemset,
                "support": support,
                "length": len(itemset),
            })

    return pd.DataFrame(frequent_itemsets), support_map, steps


def association_rules_manual(frequent_itemsets, support_map, min_confidence):
    rules = []

    for _, row in frequent_itemsets.iterrows():
        itemset = row["itemsets"]
        if len(itemset) < 2:
            continue

        for antecedent_size in range(1, len(itemset)):
            for antecedent_tuple in combinations(itemset, antecedent_size):
                antecedent = frozenset(antecedent_tuple)
                consequent = frozenset(itemset - antecedent)
                support_itemset = support_map[itemset]
                support_antecedent = support_map[antecedent]
                support_consequent = support_map[consequent]
                confidence = support_itemset / support_antecedent
                lift = confidence / support_consequent

                if confidence >= min_confidence:
                    rules.append({
                        "antecedents": antecedent,
                        "consequents": consequent,
                        "support": support_itemset,
                        "confidence": confidence,
                        "lift": lift,
                    })

    return pd.DataFrame(rules)

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
        
    st.markdown("**A. Encoding Transaksi Manual**")
    st.write("Setiap transaksi diubah menjadi tabel 1/0. Nilai 1 berarti item ada pada transaksi, nilai 0 berarti tidak ada.")
    df_encoded, unique_items = encode_transactions_manual(dataset)
    st.dataframe(df_encoded, use_container_width=True)

    st.markdown("**B. Pembentukan Kandidat dan Perhitungan Support Manual**")
    st.write("Kandidat itemset dibuat bertahap dari 1-itemset sampai 3-itemset. Support dihitung dengan menghitung jumlah transaksi yang mengandung seluruh item pada kandidat, lalu dibagi total transaksi.")
    frequent_itemsets, support_map, apriori_steps = apriori_manual(dataset, min_support, max_len=3)

    for step in apriori_steps:
        st.write(f"**C{step['size']} dan L{step['size']}**")
        if step["candidates"].empty:
            st.write("Tidak ada kandidat.")
        else:
            st.dataframe(step["candidates"], use_container_width=True)
    
    if frequent_itemsets.empty:
        st.warning("Tidak ada itemset yang memenuhi nilai Minimum Support.")
    else:
        st.markdown("**C. Frequent 1-Itemset**")
        itemset_1 = frequent_itemsets[frequent_itemsets['length'] == 1].copy()
        if not itemset_1.empty:
            itemset_1['itemsets'] = itemset_1['itemsets'].apply(format_itemset)
            st.dataframe(itemset_1[['itemsets', 'support']], use_container_width=True)
        else:
            st.write("Tidak ada.")
            
        st.markdown("**D. Frequent 2-Itemset**")
        itemset_2 = frequent_itemsets[frequent_itemsets['length'] == 2].copy()
        if not itemset_2.empty:
            itemset_2['itemsets'] = itemset_2['itemsets'].apply(format_itemset)
            st.dataframe(itemset_2[['itemsets', 'support']], use_container_width=True)
        else:
            st.write("Tidak ada.")
            
        st.markdown("**E. Frequent 3-Itemset**")
        itemset_3 = frequent_itemsets[frequent_itemsets['length'] == 3].copy()
        if not itemset_3.empty:
            itemset_3['itemsets'] = itemset_3['itemsets'].apply(format_itemset)
            st.dataframe(itemset_3[['itemsets', 'support']], use_container_width=True)
        else:
            st.write("Tidak ada.")

    # 5. Hasil perhitungan/model
    st.subheader("5. Hasil Perhitungan (Association Rules)")
    if frequent_itemsets.empty:
         st.warning("Tidak dapat menghasilkan rules karena frequent itemset kosong.")
    else:
        st.write("Rules dibuat manual dari setiap frequent itemset berukuran minimal 2. Untuk setiap rule A -> B, confidence dihitung dari support(A dan B) / support(A), lalu lift dihitung dari confidence / support(B).")
        rules = association_rules_manual(frequent_itemsets, support_map, min_confidence)
        
        if rules.empty:
            st.warning("Tidak ada association rules yang memenuhi nilai Minimum Confidence.")
        else:
            rules_show = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
            rules_show['antecedents'] = rules_show['antecedents'].apply(format_itemset)
            rules_show['consequents'] = rules_show['consequents'].apply(format_itemset)
            
            st.dataframe(rules_show, use_container_width=True)
            
            # 6. Interpretasi & 7. Kesimpulan
            st.subheader("6. Interpretasi & 7. Kesimpulan")
            
            # Mendapatkan rule terbaik berdasarkan lift tertinggi
            best_rule = rules.loc[rules['lift'].idxmax()]
            ant = format_itemset(best_rule['antecedents'])
            con = format_itemset(best_rule['consequents'])
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
