# Transformer Encoder

## Identitas
- Nama: Rafi Diva Efangga
- NIM: 123450001
- Komponen: Transformer Encoder
- Modul: 09
- Tugas tambahan: Integrator

## Deskripsi
Mengembangkan model Transformer Encoder untuk pengenalan aktivitas manusia di depan meja berdasarkan data pose sequence. Model menggunakan urutan pose yang direkam dengan frekuensi 10 FPS selama 10 detik. Satu detik pertama setiap kondisi dibuang sehingga tersisa 90 frame untuk proses pengolahan data.

## Kelas Aktivitas
1. Duduk diam
2. Mengetik
3. Menulis
4. Berdiri
5. Berjalan melintas
6. Mengambil barang dari kotak

## Rencana Pengerjaan
1. Memahami struktur dan karakteristik dataset pose.
2. Melakukan preprocessing dan normalisasi data.
3. Mengimplementasikan Temporal Patch Embedding.
4. Mengimplementasikan Positional Encoding.
5. Membangun arsitektur Transformer Encoder.
6. Menambahkan Classification Head.
7. Melakukan training, testing, dan evaluasi model.
8. Menyiapkan model untuk ekspor ONNX.

## Evaluasi Model
- F1-macro.
- Confusion Matrix.
- Rata-rata dan simpangan baku dari 3 seed.
- Ukuran model ONNX (KB).
- Latensi inferensi (ms).

## Tugas Integrator
Membantu mengintegrasikan model anggota kelompok dengan menyelaraskan format input, output, pembagian dataset, dan sistem evaluasi agar seluruh model dapat dibandingkan secara konsisten.