# Estimasi Hb dan skrining anemia dari citra konjungtiva mata

Repositori ini berisi pipeline aktif model mata, artefak hasil pelatihan, grafik evaluasi agregat, dan [laporan hasil lengkap](HASIL_MODEL_MATA.md). Dataset mentah dan foto uji tidak disertakan.

## Yang sudah dikerjakan

1. Membaca metadata Hb, jenis kelamin, dan usia dari bagian India dan Italia pada dataset *Eyes-defy-anemia*. Dari 218 subjek awal, 211 mempunyai Hb numerik dan citra konjungtiva *forniceal* yang dapat dipakai.
2. Mengekstraksi statistik warna RGB, LAB, HSV, dan luas *mask* jaringan dari citra *forniceal*. Pipeline Ridge memakai 12 fitur citra ditambah indikator jenis kelamin.
3. Membandingkan kandidat regresi Ridge, Gradient Boosting, dan Random Forest serta susunan fitur citra saja, citra + jenis kelamin, dan citra + jenis kelamin + usia. Konfigurasi akhir Ridge memakai 13 fitur dan `alpha=1.0`.
4. Memakai MobileNetV2 pralatih ImageNet sebagai ekstraktor fitur beku, lalu melatih Logistic Regression untuk probabilitas anemia. CNN tidak dilatih ulang dan tidak menghasilkan Hb.
5. Menggabungkan dua keputusan kelas dengan aturan konservatif: status akhir anemia jika salah satu model menyatakan anemia. Nilai Hb akhir tetap berasal dari Ridge.
6. Menghitung validasi OOF lima *fold* berstratifikasi pada pasien yang sama untuk kedua model, membuat tabel metrik, matriks kebingungan, serta [kurva presisi–*recall* dan F1](outputs/plots/eye_precision_recall_f1.png).
7. Mencoba inferensi pada tujuh foto lokal dalam skenario masukan jenis kelamin perempuan dan laki-laki. Foto-foto itu tidak memiliki Hb laboratorium; hasilnya hanya pemeriksaan fungsi, bukan nilai akurasi pada data baru.
8. Membuat visualisasi sebelum–sesudah berbasis warna jaringan. Visualisasi ini bukan peta atribusi MobileNetV2 dan tidak disertakan karena memuat foto mata.

## Hasil utama

Evaluasi berpasangan OOF mencakup 211 pasien: 90 anemia dan 121 normal.

| Model | MAE Hb (g/dL) | RMSE Hb (g/dL) | *r* | Akurasi | Sensitivitas anemia | F1 anemia |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 1,2805 | 1,6467 | 0,7136 | 74,88% | 73,33% | 71,35% |
| MobileNetV2 + Logistic Regression | — | — | — | 73,46% | 80,00% | 72,00% |
| Gabungan | 1,2805 | 1,6467 | 0,7136 | 72,04% | 93,33% | 74,01% |

Gabungan meningkatkan sensitivitas tetapi menambah positif palsu: 53 pada gabungan, dibanding 29 pada Ridge. MobileNetV2 tidak mempunyai MAE Hb karena keluarannya hanya probabilitas anemia. Detail metode, presisi, R², serta seluruh matriks kebingungan ada di [laporan](HASIL_MODEL_MATA.md).

## Isi repositori

| Lokasi | Isi |
|---|---|
| `feature_extraction.py`, `image_processor.py`, `data_loader.py` | Pengolahan citra dan metadata |
| `evaluation.py`, `train_final.py`, `train_mobilenet.py` | Perbandingan kandidat dan pelatihan model |
| `inference.py` | Prediksi Hb serta status akhir dari citra BGR dan jenis kelamin |
| `plot_eye_metrics.py` | Evaluasi berpasangan dan grafik metrik |
| `test_manual.py`, `visualize_model_heatmap.py` | Pemeriksaan gambar lokal dan visualisasi warna jaringan |
| `outputs/models/` | Tiga artefak `.joblib` model aktif dan metrik per *fold* CNN |
| `outputs/plots/` | Grafik agregat tanpa foto pasien |
| `outputs/test_results/` | Ringkasan keluaran pada tujuh gambar tanpa label acuan |
| `HASIL_MODEL_MATA.md` | Laporan hasil dan batasan interpretasi |

Eksperimen lama yang memakai fitur atau pembagian data berbeda tidak dimasukkan agar metriknya tidak tercampur dengan hasil akhir. Ketiga artefak model aktif berukuran sekitar 11,4 MB secara keseluruhan.

## Menjalankan inferensi

Gunakan lingkungan Python yang mendukung TensorFlow dan pasang dependensi:

```powershell
python -m pip install -r requirements.txt
```

Contoh penggunaan dari direktori repositori:

```python
import cv2
from inference import predict_from_image

image = cv2.imread("path/ke/citra_mata.png")
result = predict_from_image(image, gender="F")
print(result)
```

`gender` harus berisi jenis kelamin pasien yang benar, bukan tebakan. Input gambar ke fungsi adalah array BGR dari OpenCV. Keluaran `hgb_predicted` berasal dari Ridge; `status` adalah keputusan gabungan. Jika model CNN gagal dimuat, modul dapat kembali ke Ridge saja, sehingga periksa field `source` pada hasil. Hasil hanya untuk penelitian/skrining dan bukan pengganti tes Hb laboratorium.

## Menyiapkan data untuk pelatihan ulang

Peroleh dataset *Eyes-defy-anemia* dari sumber aslinya. Letakkan folder `India/` dan `Italy/` beserta masing-masing file Excel di `data/dataset anemia/`, atau tentukan lokasinya melalui variabel lingkungan `EYES_DEFY_ANEMIA_DATASET`. Folder `data/` diabaikan Git.

```powershell
$env:EYES_DEFY_ANEMIA_DATASET = "D:\lokasi\dataset anemia"
python evaluation.py
python train_final.py
python train_mobilenet.py
python plot_eye_metrics.py
```

Jalankan perintah dari akar repositori. `evaluation.py` menghasilkan evaluasi kandidat dengan `KFold`; `plot_eye_metrics.py` memakai pembagian berstratifikasi yang sama bagi Ridge dan MobileNetV2 untuk kurva evaluasi gabungan. Melatih ulang model akan menimpa artefak model yang disertakan, jadi gunakan salinan repositori bila ingin mempertahankan hasil yang dipublikasikan.

## Berkas yang sengaja tidak dipublikasikan

Dataset mentah berukuran sekitar 685 MB dan berisi citra pasien. Tujuh foto uji lokal serta visualisasi yang memuat foto mata juga dikecualikan karena izin publikasinya tidak terverifikasi. Notebook dan model eksperimen lama tidak menjadi dasar hasil akhir dan tidak disertakan. Jika ingin menambahkan citra atau data ke repositori publik, pastikan izin distribusi, atribusi, dan aspek privasinya terlebih dahulu.

## Mengunggah ke GitHub

Setelah membuat repositori kosong di GitHub, jalankan dari folder ini:

```powershell
git init -b main
git add .
git commit -m "Add eye anemia model and results"
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```

Ganti URL remote dengan repositori milik Anda. Folder ini belum dipush dan tidak menyertakan kredensial atau remote Git.
