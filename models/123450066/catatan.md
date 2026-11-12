
# Eksperimen RNN — Muhammad Aqil Ramadhan

**NIM:** 123450066

## 1. Deskripsi Kasus

Penelitian berfokus pada pengenalan aktivitas manusia
berdasarkan rekaman aktivitas yang dilakukan oleh partisipan.

Eksperimen dirancang menggunakan dua faktor:
- Partisipan sebagai faktor manual.
- Aktivitas sebagai faktor isyarat.

Faktor otomatis ditetapkan pada kondisi skip.

## 2. Partisipan

Eksperimen melibatkan delapan partisipan:

P01, P02, P03, P04, P05, P06, P07, dan P08.

Satu partisipan merupakan satu blok eksperimen.
Skrip berhenti setiap kali terjadi pergantian partisipan.

## 3. Kelas Aktivitas

Terdapat enam aktivitas yang ditampilkan melalui
isyarat pada layar selama 10 detik:

1. Duduk diam.
2. Mengetik.
3. Menulis.
4. Berdiri.
5. Berjalan melintas.
6. Mengambil barang dari kotak.

## 4. Model yang Dikembangkan

Implementasi mencakup tiga arsitektur:

- Simple Recurrent Neural Network (RNN).
- Long Short-Term Memory (LSTM).
- Gated Recurrent Unit (GRU).

LSTM digunakan sebagai kandidat model utama,
sedangkan RNN dan GRU menjadi model pembanding.

## 5. Rancangan Arsitektur

Konfigurasi awal:

| Parameter | Nilai |
|---|---|
| Hidden size | 64 |
| Jumlah layer | 2 |
| Bidirectional | True |
| Dropout | 0.3 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | CrossEntropyLoss |

Konfigurasi ini masih bersifat awal dan akan disesuaikan
berdasarkan karakteristik dataset primer.

## 6. Pengujian

Pengujian awal dilakukan menggunakan data sintetis
untuk memeriksa implementasi arsitektur.

Pengujian pada dataset primer belum dilakukan karena
dataset dan spesifikasi format data belum tersedia.

## 7. Evaluasi

Metrik evaluasi yang direncanakan:

- Accuracy.
- Macro-F1.
- Confusion Matrix.

## 8. Pekerjaan Selanjutnya

1. Memperoleh dataset primer.
2. Mengidentifikasi format fitur dan label.
3. Menentukan panjang sequence.
4. Menyesuaikan preprocessing.
5. Melakukan training RNN, LSTM, dan GRU.
6. Membandingkan hasil eksperimen.
7. Menjalankan evaluasi sesuai protokol tim.

## 9. Batasan

Belum tersedia hasil eksperimen pada dataset primer.
Seluruh konfigurasi model masih merupakan rancangan awal.
