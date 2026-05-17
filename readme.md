# Data Mining Learning Application

Aplikasi web interaktif berbasis **Streamlit** untuk mempelajari algoritma-algoritma utama dalam Data Mining.

## Algoritma yang Tersedia
1. Apriori (Association Rule)
2. K-Means (Clustering)
3. Regression (Prediksi)
4. Klasifikasi C4.5 (Decision Tree)
5. Naive Bayes (Probabilistik)

## Struktur Proyek
- `app.py`: Skrip utama aplikasi Streamlit.
- `requirements.txt`: Daftar dependensi library Python.
- `data/`: Folder untuk menyimpan dataset (CSV, dll).
- `utils/`: Folder untuk skrip pembantu (helper functions, visualisasi, dll).

## Cara Menjalankan Aplikasi
## Cara Menjalankan Aplikasi (Langkah yang Baik dan Benar)

Ikuti langkah-langkah berikut agar aplikasi berjalan tanpa error pada lingkungan Windows:

1. (Opsional tapi direkomendasikan) Buat dan aktifkan virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1    # PowerShell
   # atau
   .\.venv\Scripts\activate.bat    # Command Prompt
   ```

2. Pastikan `pip` up-to-date lalu instal dependensi dari `requirements.txt` menggunakan interpreter yang sama:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Verifikasi instalasi (cek versi `streamlit` dan `seaborn`):
   ```powershell
   python -c "import streamlit as st; print('streamlit', st.__version__)"
   python -c "import seaborn as sns; print('seaborn', sns.__version__)"
   ```

4. Jalankan aplikasi dengan interpreter yang sama (hindari menjalankan `streamlit` global yang berbeda):
   ```powershell
   python -m streamlit run app.py
   ```

5. Akses aplikasi di browser: http://localhost:8501 (atau alamat yang ditampilkan oleh Streamlit).

Troubleshooting singkat:
- Jika muncul `ModuleNotFoundError: No module named 'streamlit'` atau `No module named 'seaborn'`: pastikan Anda menjalankan `python -m pip install -r requirements.txt` menggunakan interpreter yang sama dengan yang menjalankan `streamlit` (gunakan `python -m streamlit run ...`).
- Jika `pip install` gagal karena file sedang digunakan (WinError 32): tutup proses `streamlit` yang sedang berjalan atau terminal yang menjalankannya; Anda dapat menghentikan proses di Task Manager atau pakai:
  ```powershell
  taskkill /IM streamlit.exe /F
  ```
- Jika ada masalah izin, gunakan `--user` atau jalankan PowerShell/CMD sebagai Administrator:
  ```powershell
  python -m pip install --user -r requirements.txt
  ```
- Jika aplikasi tidak muncul di http://localhost:8501, periksa port lain yang digunakan oleh Streamlit (output di terminal saat aplikasi berjalan) atau matikan aplikasi lain yang memakai port tersebut.

Catatan:
- Selalu gunakan `python -m pip` dan `python -m streamlit` untuk memastikan konsistensi interpreter.
- Proyek ini menguji dengan Python 3.8+; gunakan versi Python modern (mis. 3.10/3.11/3.12).

Jika Anda mau, saya bisa menambahkan skrip PowerShell kecil untuk meng-setup environment otomatis.
