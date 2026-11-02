# Catatan model 122450102

## Alasan optimizer
Optimizer yang digunakan sebagai baseline adalah Adam karena memiliki mekanisme penyesuaian learning rate secara adaptif. Pemilihan ini akan dievaluasi setelah eksperimen CNN dijalankan. Alternatif optimizer akan dipertimbangkan berdasarkan hasil validasi.

## Regularisasi
Regularisasi akan digunakan untuk mengurangi risiko overfitting. Kandidat awal adalah dropout. Pengaruhnya akan dievaluasi melalui eksperimen ablation.

## Pencarian hyperparameter (tabel)
| Hyperparameter | Kandidat |
|---|---|
| Learning rate | 0.001, 0.0001 |
| Batch size | 16, 32 |
| Jumlah filter | 16, 32, 64 |
| Kernel size | 3, 5 |
| Dropout | 0.2, 0.5 |

## Ablation (>= 2)
1. Membandingkan CNN tanpa dropout dengan CNN menggunakan dropout.
2. Membandingkan CNN dengan jumlah filter 16 dan 32.

## Hasil 3 seed konfigurasi akhir
Hasil belum tersedia karena eksperimen belum dijalankan. Nilai val_acc, val_macro_f1, dan durasi run akan dicatat untuk tiga seed pada konfigurasi akhir, serta untuk minimal dua eksperimen ablation.
