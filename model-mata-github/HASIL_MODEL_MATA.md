# Hasil Model Citra Mata

Dokumen ini merangkum hasil akhir pipeline citra mata yang aktif di proyek. Angka utama berasal dari evaluasi *out-of-fold* (OOF) Ridge Regression dan MobileNetV2 pada 211 pasien yang sama. Hasil eksperimen lama, skor pelatihan, dan contoh gambar tanpa Hb laboratorium tidak dicampurkan ke tabel kinerja tersebut.

## 1. Data yang digunakan

Sumber data adalah *Eyes-defy-anemia*, yang terdiri atas 95 subjek dari India dan 123 subjek dari Italia. Setiap subjek memiliki metadata jenis kelamin, usia, dan Hb; tersedia pula citra mata asli serta, bila ada, citra konjungtiva yang telah disegmentasi. Pipeline ini memilih citra konjungtiva *forniceal*, bukan foto mata asli. Dokumentasi dataset menyebutkan bahwa pengambilan citra memakai telepon Samsung S6 dengan perangkat pembesar dan pencahayaan LED yang distandarkan.

Satu baris Italia tidak memiliki Hb numerik yang dapat digunakan. Enam subjek Italia lain tidak memiliki citra *forniceal* karena area tersebut tidak tampak pada foto. Akhirnya, 211 pasien memiliki pasangan citra dan Hb yang dapat dievaluasi. Tidak ada pasien India yang gugur pada tahap ini.

| Kelompok | Jumlah pasien | Anemia | Normal |
|---|---:|---:|---:|
| India | 95 | 68 | 27 |
| Italia | 116 | 22 | 94 |
| **Total** | **211** | **90** | **121** |

Dalam 211 pasien tersebut terdapat 84 perempuan dan 127 laki-laki. Hb laboratorium berkisar 7,0–17,4 g/dL, dengan rata-rata 12,76 g/dL. Label anemia dalam proyek ditetapkan bila Hb <12 g/dL untuk perempuan atau <13 g/dL untuk laki-laki. Proporsi anemia berbeda cukup besar antara data India (68/95) dan Italia (22/116); karena itu, hasil validasi gabungan ini tidak boleh langsung dianggap mewakili semua lokasi atau kondisi pengambilan gambar.

Sumber data asli: dokumentasi dan metadata *Eyes-defy-anemia* (tidak disertakan dalam repositori). Pengolahan lokal dapat ditelusuri pada [pemuat metadata](data_loader.py) dan [pemilih citra](image_processor.py).

## 2. Model yang benar-benar dijalankan

### Ridge Regression untuk Hb

Ridge menerima 13 masukan: 12 statistik citra (`mean_R`, `mean_G`, `mean_B`, `mean_L`, `mean_a`, `mean_b`, `mean_S`, `mean_V`, `std_a`, `std_b`, `std_S`, `n_valid_px`) dan satu indikator jenis kelamin (`is_female`). Fitur warna dihitung setelah penolakan latar putih/hitam, pemilihan piksel jaringan berdasarkan LAB/HSV, serta pembersihan morfologi dan komponen terbesar. Fitur distandarkan dengan `StandardScaler` di dalam pipeline sebelum Ridge menghasilkan Hb dalam g/dL.

Parameter regularisasi Ridge yang dipakai adalah `alpha=1.0`. Pada skrip final parameter ini ditetapkan, bukan dipilih melalui pencarian hiperparameter yang terdokumentasi. Usia dan fitur turunan dari label tidak dipakai sebagai masukan model final. Nilai Hb prediksi di bawah ambang sesuai jenis kelamin menghasilkan status Ridge **Anemia**; selain itu **Normal**.

### MobileNetV2 untuk status anemia

MobileNetV2 pralatih ImageNet menerima citra RGB berukuran 224 × 224 piksel. Backbone dibekukan; tidak ada *fine-tuning* bobot CNN pada data mata ini. Keluaran spasial 7 × 7 × 1.280 diratakan menjadi 62.720 fitur. Setelah standardisasi, `LogisticRegression` dengan `C=1.0` dan `class_weight="balanced"` dilatih untuk memprediksi probabilitas kelas anemia. Probabilitas ≥0,5 diubah menjadi status **Anemia**. MobileNetV2 tidak menghasilkan angka Hb.

Karena klasifikasi dilatih di luar backbone, penyebutan “MobileNetV2” dalam tabel hasil berarti **ekstraktor MobileNetV2 beku + classifier Logistic Regression**, bukan CNN yang dilatih ulang dari awal. Berkas `mobilenet_anemia.joblib` menyimpan scaler dan classifier; ekstraktor disimpan terpisah di `mobilenet_extractor.joblib`.

### Aturan gabungan

Sistem mengeluarkan Hb dari Ridge saja. Untuk status akhir, sistem menetapkan **Anemia** jika Ridge **atau** MobileNetV2 menyatakan anemia. Hanya jika keduanya menyatakan normal, status akhir menjadi **Normal**. Bila kedua model berbeda, sistem memilih anemia dan menandai bahwa model tidak sepakat. Jadi, implementasi sebenarnya adalah aturan *OR* yang konservatif, bukan pemungutan suara mayoritas. Jika artefak MobileNetV2 gagal dimuat, inferensi kembali memakai Ridge saja.

Sumber implementasi: [ekstraksi fitur](feature_extraction.py), [pelatihan Ridge](train_final.py), [pelatihan MobileNetV2](train_mobilenet.py), dan [inferensi gabungan](inference.py). Artefak yang disertakan berada di direktori `outputs/models`.

## 3. Cara evaluasi

Evaluasi perbandingan dilakukan dengan *5-fold StratifiedKFold*, `shuffle=True`, dan `random_state=42`, memakai label anemia untuk stratifikasi. Setiap pasien diprediksi sekali oleh model yang dilatih tanpa pasien tersebut. Prediksi lima *fold* digabungkan sebagai prediksi OOF. Ridge dan MobileNetV2 menggunakan pasien serta pembagian *fold* yang sama. Scaler Ridge, scaler fitur CNN, dan classifier Logistic Regression hanya dilatih pada bagian latih tiap *fold*; ekstraktor MobileNetV2 tetap beku.

Regresi Hb dinilai dengan MAE, RMSE, korelasi Pearson (*r*), dan R². Klasifikasi anemia dinilai dengan akurasi, presisi, sensitivitas (*recall*), F1-score, dan matriks kebingungan. Kelas positif adalah anemia. Untuk sistem gabungan, metrik regresi tetap mengukur Hb dari Ridge; metrik klasifikasi mengukur status akhir hasil aturan *OR*.

Perlu dibedakan dari [laporan evaluasi kandidat](outputs/evaluation_report.json) dan metrik yang tersimpan dalam artefak Ridge: keduanya menggunakan `KFold` teracak, bukan pembagian *stratified* berpasangan yang digunakan untuk tabel akhir Ridge–MobileNetV2 di bawah ini. Karena pembagian *fold* berbeda, angka 1,2962 g/dL pada laporan kandidat tidak sama dengan MAE 1,2805 g/dL pada evaluasi akhir. Ini perbedaan rancangan validasi, bukan perubahan satuan atau pembulatan. Prosedur evaluasi berpasangan dapat dilihat pada [skrip kurva metrik](plot_eye_metrics.py); hasil per *fold* MobileNetV2 tersimpan dalam [metrik MobileNetV2](outputs/models/mobilenet_cv_metrics.json).

## 4. Hasil estimasi Hb

**Tabel 4.X. Performa estimasi Hb pada tingkat pasien (*out-of-fold*, 211 pasien)**

| Model | MAE (g/dL) | RMSE (g/dL) | *r* Pearson | R² |
|---|---:|---:|---:|---:|
| Ridge Regression | 1,2805 | 1,6467 | 0,7136 | 0,5088 |
| MobileNetV2 | — | — | — | — |
| Gabungan Ridge + MobileNetV2 | 1,2805 | 1,6467 | 0,7136 | 0,5088 |

Rata-rata selisih absolut Hb prediksi Ridge terhadap Hb laboratorium adalah 1,2805 g/dL. RMSE yang lebih besar, 1,6467 g/dL, menunjukkan adanya beberapa kesalahan yang lebih besar daripada galat rata-rata. Korelasi *r* sebesar 0,7136 menunjukkan hubungan positif yang cukup kuat, tetapi tidak berarti setiap prediksi dekat dengan nilai laboratorium; kedekatan nilainya tetap dinilai dari MAE dan RMSE.

Baris “gabungan” mempunyai angka regresi yang identik dengan Ridge karena sistem tidak menggabungkan dua prediksi Hb. Memberi MAE atau RMSE tersendiri kepada MobileNetV2 akan keliru: keluarannya adalah probabilitas anemia, bukan Hb. Evaluasi ini bersifat internal OOF, belum merupakan pengujian pada kohort eksternal berlabel Hb.

## 5. Hasil klasifikasi anemia

**Tabel 4.X. Performa klasifikasi anemia pada tingkat pasien (*out-of-fold*, 211 pasien)**

| Model | Akurasi | Presisi | Sensitivitas | F1-score | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge Regression | 74,88% | 69,47% | 73,33% | 71,35% | 92 | 29 | 24 | 66 |
| MobileNetV2 | 73,46% | 65,45% | 80,00% | 72,00% | 83 | 38 | 18 | 72 |
| Gabungan Ridge + MobileNetV2 | 72,04% | 61,31% | 93,33% | 74,01% | 68 | 53 | 6 | 84 |

TN adalah pasien normal yang diprediksi normal; FP pasien normal yang diprediksi anemia; FN pasien anemia yang diprediksi normal; TP pasien anemia yang diprediksi anemia. Matriks kebingungan masing-masing model, dengan baris **aktual normal, aktual anemia** dan kolom **prediksi normal, prediksi anemia**, adalah:

| Model | Matriks kebingungan |
|---|---|
| Ridge | `[[92, 29], [24, 66]]` |
| MobileNetV2 | `[[83, 38], [18, 72]]` |
| Gabungan | `[[68, 53], [6, 84]]` |

Ridge mengenali 66 dari 90 pasien anemia. MobileNetV2 mengenali 72 pasien, tetapi menghasilkan lebih banyak positif palsu. Aturan gabungan mengenali 84 pasien anemia dan menyisakan enam negatif palsu. Dibanding Ridge saja, sensitivitas gabungan naik dari 73,33% menjadi 93,33%, tetapi positif palsu naik dari 29 menjadi 53 pasien; presisi dan akurasi turun. Dari 211 pasien, kedua model memberi status yang sama pada 142 pasien dan berbeda pada 69 pasien. Dengan dua model, perbedaan tersebut selalu diputuskan sebagai anemia oleh aturan akhir.

Konsekuensinya, pilihan gabungan masuk akal **jika tujuan utama adalah skrining yang mengurangi anemia terlewat**. Hasil ini tidak membuktikan bahwa gabungan lebih baik pada semua kriteria. Biaya positif palsu dan kebutuhan konfirmasi Hb laboratorium harus tetap dilaporkan.

## 6. Alasan konfigurasi akhir

Dalam [perbandingan kandidat](outputs/evaluation_report.json), Ridge dengan 12 fitur citra + jenis kelamin memperoleh MAE 1,2962 g/dL pada skema `KFold` kandidat. Menambahkan usia menurunkannya menjadi 1,2944 g/dL, hanya selisih 0,0018 g/dL, sementara *recall* klasifikasi turun dari 73,33% menjadi 72,22%. Karena perbaikan MAE amat kecil, konfigurasi 13 fitur tanpa usia dipertahankan. Dibandingkan fitur citra saja, penambahan jenis kelamin menurunkan MAE kandidat dari 1,4454 menjadi 1,2962 g/dL. Pemilihan `alpha=1.0` pada Ridge dan ambang probabilitas 0,5 pada MobileNetV2 merupakan konfigurasi yang dipakai di kode; dokumen ini tidak mengklaim keduanya sebagai hiperparameter optimal hasil pencarian menyeluruh.

Peran setiap komponen berbeda: Ridge menyediakan besaran Hb yang dapat dibandingkan dengan laboratorium; MobileNetV2 menyediakan penilaian kelas anemia dari pola citra; aturan gabungan menentukan status akhir yang lebih sensitif. Tidak ada langkah yang mengubah probabilitas MobileNetV2 menjadi angka Hb. Parameter `confidence` yang dikeluarkan modul inferensi adalah skor gabungan buatan dari margin Hb dan probabilitas classifier, bukan ukuran kepastian klinis yang sudah dikalibrasi.

## 7. Visualisasi yang tersedia

1. [Kurva presisi–*recall* dan F1 terhadap ambang probabilitas](outputs/plots/eye_precision_recall_f1.png). Bagian kiri membandingkan Ridge, MobileNetV2, dan aturan gabungan pada prediksi OOF. Bagian kanan menunjukkan perubahan F1 ketika ambang probabilitas MobileNetV2 diubah; garis vertikal menandai ambang 0,5 yang digunakan. Kurva gabungan tidak setara dengan skor probabilitas Hb: pada setiap ambang, statusnya tetap berasal dari aturan *OR* terhadap status Ridge.
2. Visualisasi sebelum–sesudah pada tujuh gambar uji dibuat menggunakan [skrip visualisasi](visualize_model_heatmap.py), tetapi gambar yang memuat foto mata tidak disertakan dalam repositori. Peta merah-biru dihitung dari warna LAB/HSV dan *mask* jaringan. Ini **bukan** Grad-CAM, *saliency map*, atau atribusi matematis keputusan MobileNetV2. Status pada judul gambar memang berasal dari model tersimpan, tetapi area merah pada visualisasi tidak dapat ditafsirkan sebagai area yang pasti dipakai classifier untuk mengambil keputusan.
3. [Distribusi Hb menurut negara dan jenis kelamin](outputs/plots/hgb_distribution_by_country_gender.png) dan [hubungan kanal warna dengan Hb](outputs/plots/rgb_hgb_correlations.png) adalah visualisasi deskriptif dataset, bukan bukti performa prediksi eksternal.

## 8. Pemeriksaan pada gambar baru

Tujuh citra uji lokal telah dicoba pada alur inferensi; berkas citranya tidak disertakan dalam repositori. Pengujian dilakukan dalam dua skenario masukan jenis kelamin, perempuan dan laki-laki, karena jenis kelamin sebenarnya tidak disediakan. Semua gambar menghasilkan keluaran dalam pemeriksaan tersebut. Rentang Hb prediksi Ridge adalah 10,04–15,26 g/dL pada skenario perempuan dan 11,79–17,02 g/dL pada skenario laki-laki. Status gabungan yang keluar sama pada kedua skenario: enam citra berstatus anemia dan satu (`test_image03`) normal.

Angka dan status ini hanya membuktikan bahwa alur inferensi dapat dijalankan pada tujuh berkas tersebut. Ketujuh gambar tidak mempunyai Hb laboratorium atau label anemia terverifikasi, sehingga **tidak boleh digunakan untuk menghitung MAE, akurasi, sensitivitas, ataupun klaim generalisasi**. Jenis kelamin harus diisi dari informasi pasien sebenarnya ketika sistem dipakai; hasil dari jenis kelamin yang diasumsikan dapat mengubah angka Hb dan interpretasi ambangnya. Rincian keluaran tersedia pada [skenario perempuan](outputs/test_results/skenario_F.csv) dan [skenario laki-laki](outputs/test_results/skenario_M.csv).

## 9. Batasan dan pemisahan dari eksperimen lama

- Validasi utama adalah OOF pada satu kumpulan data, bukan uji prospektif atau uji pada pasien baru dengan Hb laboratorium. Ukuran data 211 pasien masih terbatas, dan komposisi anemia berbeda antara India dan Italia.
- Citra dataset diperoleh dengan perangkat dan pencahayaan yang distandarkan. Kinerja pada foto ponsel tanpa kondisi tersebut belum terukur dalam evaluasi berlabel.
- Skor probabilitas MobileNetV2 dapat sangat mendekati 0 atau 1 pada sebagian gambar; angka itu tidak boleh langsung ditafsirkan sebagai kepastian klinis atau bukti kalibrasi probabilitas.
- Notebook dan ringkasan keluaran lama tidak disertakan karena memakai rancangan fitur, ambang, atau pembagian data yang berbeda. Misalnya, angka MAE 1,09 g/dL dalam ringkasan lama bukan hasil validasi berpasangan Ridge–MobileNetV2 yang dilaporkan di sini. Angka tersebut tidak digabungkan dengan tabel akhir.
- Hasil akhir ini menggambarkan performa skrining penelitian. Status prediksi, termasuk ketika model sepakat, tetap memerlukan konfirmasi dengan pemeriksaan Hb laboratorium bila digunakan untuk keputusan kesehatan.

## Kesimpulan hasil

Pada 211 pasien yang memiliki citra *forniceal* dan Hb, Ridge menghasilkan estimasi Hb dengan MAE 1,2805 g/dL dan korelasi Pearson 0,7136. MobileNetV2 menambah jalur klasifikasi citra, bukan jalur estimasi Hb. Gabungan keduanya meningkatkan sensitivitas pendeteksian anemia menjadi 93,33% dengan enam kasus anemia terlewat, tetapi menurunkan presisi menjadi 61,31% dan menambah positif palsu menjadi 53 kasus. Itulah kelebihan sekaligus biaya nyata konfigurasi akhir yang dipilih.
