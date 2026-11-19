
# MatrixX - Methodology Anggota 1

## 1. Ruang Lingkup

Anggota 1 bertanggung jawab atas implementasi dan
pengujian perkalian matriks menggunakan dua pendekatan:

1. Sequential
2. OpenMP

Pengujian dilakukan untuk menganalisis perbedaan
performa antara eksekusi berurutan dan paralel.

## 2. Algoritma

Operasi yang digunakan adalah perkalian dua matriks
berukuran N x N.

Setiap elemen matriks hasil C dihitung menggunakan:

C[i][j] = sum(A[i][k] * B[k][j])

Implementasi menggunakan perulangan i-k-j untuk
memanfaatkan lokalitas memori.

## 3. Implementasi Sequential

Pada pendekatan Sequential, operasi perkalian matriks
dijalankan secara berurutan menggunakan satu thread.

Program dikembangkan dalam C++ dan menggunakan
std::chrono untuk mengukur waktu eksekusi.

## 4. Implementasi OpenMP

OpenMP digunakan untuk memparalelkan proses komputasi
perkalian matriks.

Direktif parallel dan for digunakan untuk membagi
pekerjaan berdasarkan indeks baris matriks.

Penjadwalan menggunakan schedule(static).

Konfigurasi pengujian meliputi:
- 1 thread
- 2 thread
- 4 thread
- 8 thread

## 5. Lingkungan Eksperimen

Prosesor: Intel Core i5-12450H
Lingkungan: Ubuntu melalui WSL2
Logical CPU tersedia: 12
RAM tersedia untuk WSL2: 7.6 GiB
Compiler: G++ 15.2.0
Bahasa: C++17
Python: 3.14.4
Matplotlib: 3.10.7

## 6. Konfigurasi Pengujian

Ukuran matriks:
- 512 x 512
- 1024 x 1024
- 2048 x 2048
- 4096 x 4096

Benchmark awal menggunakan 5 pengulangan.
Validasi tambahan menggunakan 10 pengulangan
untuk ukuran 1024 dan 4096.

## 7. Parameter Evaluasi

Execution Time:
Waktu yang dibutuhkan untuk menyelesaikan komputasi.

Speedup:
S = T_seq / T_parallel

Parallel Efficiency:
E = S / P

Keterangan:
S = Speedup
E = Efisiensi paralel
P = Jumlah thread

## 8. Validasi

Validasi dilakukan dengan membandingkan checksum
hasil Sequential dan OpenMP.

Seluruh checksum pada pengujian yang telah dilakukan
menunjukkan nilai yang sama untuk ukuran matriks
yang bersesuaian.

## 9. Batasan Eksperimen

Pengujian dilakukan pada lingkungan WSL2.
Hasil waktu eksekusi dapat dipengaruhi oleh kondisi
sistem, penggunaan CPU, cache, dan sumber daya memori.

Hasil benchmark awal dan validasi disimpan secara
terpisah agar tidak mencampurkan kelompok data.
