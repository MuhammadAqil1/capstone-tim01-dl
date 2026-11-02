# Transformer Encoderb(123450001-Rafi diva efangga)

## Alasan optimizer

Optimizer yang dipilih adalah AdamW karena memiliki mekanisme adaptif dalam memperbarui bobot model sehingga dapat membantu proses pelatihan Transformer Encoder menjadi lebih stabil. AdamW juga memisahkan weight decay dari proses pembaruan gradien sehingga membantu mengendalikan kompleksitas model dan mengurangi risiko overfitting.

## Regularisasi

Regularisasi dilakukan untuk mengurangi risiko overfitting selama proses pelatihan model. Teknik yang direncanakan adalah penggunaan Dropout pada bagian Transformer Encoder dan Classification Head, serta Weight Decay pada optimizer AdamW. Nilai Dropout dan Weight Decay akan diuji melalui pencarian hyperparameter untuk memperoleh konfigurasi yang memberikan hasil validasi terbaik.

## Pencarian hyperparameter (tabel)

Pencarian hyperparameter dilakukan untuk memperoleh kombinasi parameter yang sesuai dengan karakteristik data pose sequence.

| Parameter | Kandidat nilai |
|---|---|
| Learning Rate | 0.0001, 0.0003, 0.0005 |
| Batch Size | 16, 32 |
| Dropout | 0.1, 0.2, 0.3 |
| Weight Decay | 0.0001, 0.001, 0.01 |
| Embedding Dimension | 64, 128 |
| Attention Heads | 4, 8 |
| Encoder Layers | 2, 3, 4 |
| Temporal Patch Size | 5, 10, 20 frame |

Kombinasi hyperparameter akan dievaluasi menggunakan data validasi dengan mempertimbangkan nilai F1-macro. Konfigurasi terbaik dipilih berdasarkan performa validasi dan efisiensi model.

## Ablation (>= 2)

Eksperimen ablation dilakukan untuk mengetahui pengaruh setiap komponen terhadap performa Transformer Encoder.

1. **Dengan dan tanpa Positional Encoding**

   Membandingkan performa model yang menggunakan Positional Encoding dengan model tanpa Positional Encoding untuk mengetahui pengaruh informasi posisi terhadap pengenalan pola urutan gerakan.

2. **Temporal Patch Embedding dan Frame-wise Embedding**

   Membandingkan model yang menggunakan Temporal Patch Embedding dengan model yang memproses setiap frame sebagai token untuk mengetahui pengaruh pembentukan patch temporal terhadap performa klasifikasi.

3. **Dengan dan tanpa Dropout**

   Membandingkan performa model menggunakan Dropout dan tanpa Dropout untuk mengetahui pengaruh regularisasi terhadap kemampuan generalisasi model.

Seluruh eksperimen menggunakan pembagian data dan prosedur evaluasi yang konsisten agar hasil perbandingan dapat dipertanggungjawabkan.

## Hasil 3 seed konfigurasi akhir

Evaluasi konfigurasi akhir dilakukan menggunakan tiga seed berbeda untuk mengetahui kestabilan performa model. Hasil pengujian akan dicatat menggunakan nilai validation accuracy, F1-macro, dan durasi pelatihan.

| Seed | Validation Accuracy | Validation F1-macro | Durasi (detik) |
|---|---|---|---|
| 1 | Belum diuji | Belum diuji | Belum diuji |
| 2 | Belum diuji | Belum diuji | Belum diuji |
| 3 | Belum diuji | Belum diuji | Belum diuji |
| Rata-rata ± SD | Belum tersedia | Belum tersedia | Belum tersedia |

**Evaluasi efisiensi model:**

| Parameter | Hasil |
|---|---|
| Ukuran model ONNX (KB) | Belum diuji |
| Latensi inferensi (ms) | Belum diuji |
| Perangkat pengujian | Laptop |

Hasil akhir akan diperbarui setelah proses training, evaluasi tiga seed, dan ekspor model ONNX selesai dilakukan.