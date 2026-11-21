# Metodologi Eksperimen Hybrid MPI-OpenMP - Anggota 3

## 1. Ruang Lingkup dan Tanggung Jawab

Dokumen ini menjelaskan implementasi, pengukuran, validasi, dan analisis
perkalian matriks **Hybrid MPI-OpenMP** (Anggota 3) pada proyek MatrixX.
Pekerjaan Anggota 1 (Sequential dan OpenMP) dan Anggota 2 (MPI) digunakan
sebagai acuan standar data dan algoritma. Source code keduanya **tidak
diubah**; keduanya hanya dikompilasi dan dijalankan ulang di laptop Anggota 3
untuk perbandingan yang adil pada mesin yang sama.

## 2. Algoritma dan Standar Data

Operasi: `C[i][j] = sum(A[i][k] * B[k][j])`, tipe data `double`, matriks
disimpan row-major dalam `vector<double>` 1D (`A[i*N + j]`).

Pola data (sama dengan Anggota 1 dan 2):

```cpp
A[i][j] = ((i + j) % 100) * 0.01;
B[i][j] = ((i * 3 + j) % 100) * 0.01;
```

Loop komputasi memakai urutan `i-k-j`, sama dengan Anggota 1 dan 2. Ukuran
matriks: 512, 1024, 2048, dan 4096.

## 3. Implementasi Hybrid (`src/hybrid.cpp`)

**Inisialisasi.** `MPI_Init_thread` dengan `MPI_THREAD_FUNNELED`: hanya thread
utama yang memanggil fungsi MPI, sedangkan OpenMP dipakai hanya di dalam
komputasi lokal. `omp_set_dynamic(0)` dipakai agar jumlah thread tetap sesuai
`OMP_NUM_THREADS`; jumlah thread aktual dicatat lewat `omp_get_num_threads()`.

**Dekomposisi data (tingkat MPI).** Baris matriks A dibagi ke proses MPI.
Jika `N` tidak habis dibagi jumlah proses, `N mod P` proses pertama menerima
satu baris tambahan.

- `MPI_Scatterv`: membagikan baris A ke setiap proses.
- `MPI_Bcast`: mengirim matriks B lengkap ke semua proses (setiap proses
  menyimpan satu salinan penuh B).
- `MPI_Gatherv`: mengumpulkan potongan C ke rank 0.
- Matriks A dan C penuh hanya dialokasikan di rank 0.

**Paralelisasi lokal (tingkat OpenMP).**

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < my_rows; i++)
    for (int k = 0; k < N; k++) {
        double temp = A_local[i*N + k];
        for (int j = 0; j < N; j++)
            C_local[i*N + j] += temp * B[k*N + j];
    }
```

**Keamanan thread.** Loop yang dibagi adalah loop `i` (baris lokal). Setiap
thread menulis ke baris `C_local` yang berbeda dan hanya membaca `A_local` dan
`B`, sehingga tidak ada dua thread yang menulis ke elemen yang sama. Karena
urutan penjumlahan indeks `k` untuk setiap `C[i][j]` tidak berubah, hasil
secara teori identik dengan versi Sequential.

## 4. Lingkungan Eksperimen

| Komponen | Spesifikasi (dari `results/system_info.txt` dan BIOS) |
|---|---|
| Laptop | Acer Aspire AL14-37P |
| CPU | Intel N150, 4 core, 1 thread per core (4 logical CPU) |
| RAM | 8192 MB fisik (BIOS); 3,7 GiB terlihat di WSL2; swap 1,0 GiB |
| Sistem operasi | Windows 11 Home 26H2 (build 26300.9974) dengan WSL2 |
| Linux | Ubuntu 26.04.1 LTS, kernel 6.18.58.1-microsoft-standard-WSL2 |
| Compiler | g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0, C++17 |
| MPI | Open MPI 5.0.10 |
| Python | 3.14.4 (script benchmark dan grafik, Matplotlib) |

Perbedaan dengan laptop anggota lain: Anggota 1 memakai Intel Core i5-12450H
(12 logical CPU) dan Anggota 2 memakai Open MPI 4.1.6 pada Ubuntu 24.04.
Karena itu **angka waktu antaranggota tidak boleh dibandingkan langsung**.
Semua perbandingan antarmetode di dokumen ini memakai pengukuran ulang di laptop
Anggota 3.

Flag kompilasi (sama dengan Anggota 1 dan 2, ditambah `-fopenmp`):

```bash
mpicxx -O3 -march=native -std=c++17 -fopenmp anggota3/src/hybrid.cpp -o anggota3/hybrid
```

## 5. Metodologi Pengukuran

**Waktu eksekusi.** `MPI_Wtime()` dengan `MPI_Barrier` sebelum dan sesudah
bagian yang diukur. Bagian terukur mencakup `Scatterv`, `Bcast`, komputasi, dan
`Gatherv`, sama dengan implementasi MPI Anggota 2. Inisialisasi matriks tidak
termasuk. Waktu komputasi terlama antarproses diukur terpisah (`MPI_Reduce` dengan
`MPI_MAX`, di luar timer total), dan selisih terhadap waktu total disebut
**overhead non-komputasi**. Overhead ini berisi komunikasi MPI, sinkronisasi di
barrier, dan ketidakseimbangan beban antarproses; bukan komunikasi murni.

Perbedaan dengan Sequential dan OpenMP Anggota 1: keduanya hanya mengukur
komputasi (`steady_clock`), sedangkan Hybrid dan MPI mengukur komputasi
ditambah distribusi dan pengumpulan data.

**Binding.** Hybrid dijalankan dengan `mpirun --bind-to none` agar thread OpenMP
tidak terkunci ke satu core. OMP_NUM_THREADS diatur per konfigurasi. Tidak
memakai `--oversubscribe`: semua konfigurasi utama memiliki total thread tidak
lebih dari 4 core fisik.

**Kondisi pengukuran.** Charger terpasang, mode daya *Best performance*, tidak
ada aplikasi berat lain, dan satu putaran pemanasan dibuang di awal.
Laptop ini ternyata sangat dipengaruhi kondisi sesi (lihat bagian 9 dan 11).
Pengukuran Sequential 512 pertama bahkan mencatat 0,232 s, sedangkan pengukuran
berikutnya 0,038-0,044 s. Karena itu eksperimen dirancang berlapis.

## 6. Rancangan Eksperimen

Tiga eksperimen, semuanya di laptop yang sama:

1. **Benchmark utama** (`src/benchmark.py`): Hybrid dengan enam kombinasi
   proses × thread (1×1, 1×2, 1×4, 2×1, 2×2, 4×1; total thread ≤ 4) pada empat
   ukuran, 5 pengulangan per konfigurasi. Baseline Sequential lokal diukur per
   ukuran sebelum blok Hybrid: median dari 3 run terpisah (masing-masing 5
   pengulangan) untuk 512-2048, dan **hanya 1 run** untuk 4096 karena
   waktu. Hasil: `results/results_hybrid.csv`, `hybrid_metrics.csv`,
   `results_sequential_local.csv`.
2. **Pembanding lokal** (`src/benchmark_compare.py`): OpenMP (1, 2, 4 thread)
   dan MPI (1, 2, 4 proses) dari Anggota 1 dan 2, 5 pengulangan, memakai
   baseline yang sama. Hasil: `results/comparison_local.csv`.
3. **Uji bergantian** (`src/benchmark_interleaved.py`): Sequential, OpenMP 4
   thread, MPI 4 proses, Hybrid 1×4, dan Hybrid 2×2 dijalankan selang-seling
   dalam satu sesi dengan urutan diputar tiap putaran, 3 pengulangan per
   pengukuran. Ukuran 1024 dan 2048: 3 putaran; ukuran 4096: 2 putaran. Hasil:
   `results/interleaved_comparison.csv`, `interleaved_summary.csv`, dan
   `results/interleaved_4096/`.

Uji bergantian ditambahkan karena benchmark utama dan pembanding lokal
diukur pada sesi berbeda, dan kondisi laptop berubah antarsesi.

## 7. Validasi

Checksum (jumlah seluruh elemen C) dibandingkan dengan checksum Sequential
lokal pada ukuran yang sama dengan toleransi relatif 1e-12. Seluruh konfigurasi
pada tiga eksperimen berstatus cocok (kolom `checksum_match`). Nilai referensi:

| Ukuran | Checksum Sequential |
|---|---|
| 512 | 32863525.795196988 |
| 1024 | 263006890.099257082 |
| 2048 | 2104662477.597485304 |
| 4096 | 16838019066.392269135 |

Nilai ini sama dengan hasil Anggota 1. Pada pengujian manual ukuran 512, checksum
Hybrid identik hingga seluruh digit yang dicetak. Keterbatasan validasi:
**perbandingan elemen per elemen terhadap hasil Sequential tidak dilakukan**;
validasi hanya memakai checksum.

## 8. Metrik

```text
Speedup    = T_sequential_lokal / T_hybrid
Efficiency = (Speedup / P) * 100%
P          = proses MPI x thread OpenMP per proses (= total thread)
```

Contoh: konfigurasi 2×2 memiliki P = 4. P dipilih sebagai total thread karena
jumlah core fisik laptop (4) adalah batas sumber daya yang dibandingkan.

Metrik tambahan di `hybrid_metrics.csv`:

- `speedup_min`: speedup berdasarkan waktu minimum, lebih tahan noise.
- `speedup_vs_hybrid1x1` dan `efficiency_vs_hybrid1x1_percent`: speedup
  relatif terhadap Hybrid 1×1 pada ukuran yang sama. Metrik ini memisahkan
  skalabilitas dari selisih kode pada satu thread, tetapi **bergantung pada
  satu-satunya referensi 1×1 yang noisy**, sehingga nilainya bisa melebihi 100%
  tanpa menandakan speedup superlinear yang nyata.

## 9. Hasil

### 9.1 Benchmark utama Hybrid (rata-rata 5 pengulangan)

Baseline Sequential lokal (median): 0,0502 s (512), 0,6300 s (1024),
6,2501 s (2048), dan 53,3248 s (4096; satu run saja).

| Ukuran | MPI×OMP | P | Waktu (s) | Speedup | Efisiensi (%) | Overhead non-komputasi (% dari total) |
|---|---|---|---|---|---|---|
| 512 | 1×1 | 1 | 0,0524 | 0,96 | 95,8 | 1,31 |
| 512 | 1×2 | 2 | 0,0230 | 2,18 | 109,2 | 2,77 |
| 512 | 1×4 | 4 | 0,0149 | 3,36 | 84,1 | 4,47 |
| 512 | 2×1 | 2 | 0,0407 | 1,23 | 61,7 | 3,99 |
| 512 | 2×2 | 4 | 0,0221 | 2,27 | 56,7 | 6,14 |
| 512 | 4×1 | 4 | 0,0549 | 0,91 | 22,8 | 4,55 |
| 1024 | 1×1 | 1 | 0,8685 | 0,73 | 72,5 | 0,49 |
| 1024 | 1×2 | 2 | 0,3429 | 1,84 | 91,9 | 0,56 |
| 1024 | 1×4 | 4 | 0,2738 | 2,30 | 57,5 | 1,42 |
| 1024 | 2×1 | 2 | 0,7059 | 0,89 | 44,6 | 0,70 |
| 1024 | 2×2 | 4 | 0,4157 | 1,52 | 37,9 | 1,76 |
| 1024 | 4×1 | 4 | 0,5671 | 1,11 | 27,8 | 3,03 |
| 2048 | 1×1 | 1 | 9,5468 | 0,65 | 65,5 | 0,26 |
| 2048 | 1×2 | 2 | 4,9143 | 1,27 | 63,6 | 0,18 |
| 2048 | 1×4 | 4 | 4,1823 | 1,49 | 37,4 | 0,30 |
| 2048 | 2×1 | 2 | 6,2320 | 1,00 | 50,1 | 0,49 |
| 2048 | 2×2 | 4 | 3,6589 | 1,71 | 42,7 | 0,63 |
| 2048 | 4×1 | 4 | 4,2773 | 1,46 | 36,5 | 0,94 |
| 4096 | 1×1 | 1 | 63,0056 | 0,85 | 84,6 | 0,31 |
| 4096 | 1×2 | 2 | 45,7310 | 1,17 | 58,3 | 0,89 |
| 4096 | 1×4 | 4 | 37,6786 | 1,42 | 35,4 | 0,49 |
| 4096 | 2×1 | 2 | 50,2094 | 1,06 | 53,1 | 0,42 |
| 4096 | 2×2 | 4 | 39,7147 | 1,34 | 33,6 | 0,40 |
| 4096 | 4×1 | 4 | 41,2950 | 1,29 | 32,3 | 0,50 |

**Peringatan pembacaan.** Speedup 4096 pada tabel ini dihitung dari baseline
satu run yang 12,5% lebih lambat daripada Sequential pada uji bergantian
(53,32 s vs 47,40 s), sehingga cenderung terlalu tinggi. Untuk 4096 gunakan
tabel 9.2.

### 9.2 Uji bergantian lintas metode (4 unit pemrosesan)

Median rata-rata antarputaran; rentang = minimum-maksimum antarputaran;
speedup dan efisiensi terhadap Sequential pada sesi yang sama.

| Ukuran | Metode | Median (s) | Rentang (s) | Speedup | Efisiensi (%) |
|---|---|---|---|---|---|
| 1024 (3 putaran) | Sequential | 0,5725 | 0,5701-0,6540 | 1,00 | 100 |
| 1024 | OpenMP 4 thread | 0,1998 | 0,1997-0,2474 | 2,86 | 71,6 |
| 1024 | Hybrid 1×4 | 0,2060 | 0,2041-0,2065 | 2,78 | 69,5 |
| 1024 | Hybrid 2×2 | 0,3262 | 0,3194-0,3453 | 1,75 | 43,9 |
| 1024 | MPI 4 proses | 0,4843 | 0,4823-0,5492 | 1,18 | 29,5 |
| 2048 (3 putaran) | Sequential | 5,5260 | 5,3779-5,5900 | 1,00 | 100 |
| 2048 | OpenMP 4 thread | 2,8808 | 2,8347-2,9146 | 1,92 | 48,0 |
| 2048 | Hybrid 1×4 | 2,9386 | 2,9360-2,9861 | 1,88 | 47,0 |
| 2048 | Hybrid 2×2 | 3,7474 | 3,7194-4,1571 | 1,47 | 36,9 |
| 2048 | MPI 4 proses | 4,1844 | 4,1761-4,1917 | 1,32 | 33,0 |
| 4096 (2 putaran) | Sequential | 47,3992 | 47,3823-47,4161 | 1,00 | 100 |
| 4096 | OpenMP 4 thread | 36,6387 | 36,1557-37,1218 | 1,29 | 32,3 |
| 4096 | Hybrid 1×4 | 39,1707 | 36,6222-41,7193 | 1,21 | 30,3 |
| 4096 | Hybrid 2×2 | 39,6318 | 38,0243-41,2393 | 1,20 | 29,9 |
| 4096 | MPI 4 proses | 41,2698 | 39,3784-43,1612 | 1,15 | 28,7 |

## 10. Analisis Hasil

**Pengaruh jumlah thread (1×N).** Waktu turun saat thread bertambah, tetapi
efisiensinya turun. Pada 4096 (benchmark utama), 1×1 → 1×2 → 1×4 menghasilkan
63,0 s → 45,7 s → 37,7 s, yaitu percepatan 1,38× dan 1,67× terhadap Hybrid 1×1
dengan 2 dan 4 thread. Pola serupa muncul di ukuran lain.

**Pengaruh jumlah proses MPI.** Dengan total thread yang sama, lebih banyak
proses MPI berarti lebih lambat. Rasio waktu 4×1 terhadap 1×4 adalah 3,68×
(512), 2,07× (1024), 1,02× (2048), dan 1,10× (4096). Pada uji bergantian, 2×2
lebih lambat daripada 1×4 sebesar 58% (1024), 28% (2048), dan 1% (4096).
Pengaruh jumlah proses mengecil seiring ukuran matriks.

**Overhead komunikasi dan sinkronisasi.** Overhead non-komputasi hanya 0,18-6,14%
dari waktu total, dan di ukuran 2048 dan 4096 tidak lebih dari 0,94%. Jadi
selisih antarkombinasi di ukuran besar hampir seluruhnya berasal dari waktu
**komputasi**, bukan dari komunikasi. Satu hipotesis yang konsisten dengan pola
ini (belum diuji): setiap proses MPI memegang salinan penuh matriks B, sehingga
beberapa proses menghasilkan lebih banyak lalu lintas memori dan tekanan cache
daripada beberapa thread yang berbagi satu B. Pembuktiannya memerlukan
pengukuran cache atau counter perangkat keras yang di luar lingkup proyek ini.

**Kombinasi terbaik berdasarkan data.** Hybrid 1×4 memiliki speedup rata-rata
tertinggi di 512, 1024, dan 4096, dan waktu minimum terbaik di keempat ukuran.
Di ukuran 2048 pada benchmark utama, 2×2 tampak lebih cepat (1,71× vs 1,49×),
tetapi uji bergantian menunjukkan sebaliknya (1×4: 2,94 s; 2×2: 3,75 s).
Peringkat di 2048 pada benchmark utama karena itu dianggap artefak variasi
antarsesi. Kesimpulan: **1×4 adalah kombinasi terbaik pada laptop ini.**

**Perbandingan antarmetode (uji bergantian).**

- OpenMP 4 thread dan Hybrid 1×4 setara: selisih median 3% (1024), 2% (2048),
  dan 7% (4096), dengan rentang yang tumpang tindih. Hasil ini wajar karena
  Hybrid 1×4 pada dasarnya OpenMP dalam satu proses MPI.
- MPI 4 proses paling lambat di 1024 dan 2048: 2,42× lebih lambat daripada
  OpenMP di 1024 dan 1,45× di 2048.
- Di 4096 selisih antarmetode menyempit menjadi 12,6% antara tercepat
  (OpenMP) dan terlambat (MPI), dan rentang keempat metode saling tumpang
  tindih. Dengan hanya 2 putaran, **tidak ada metode yang dapat dinyatakan lebih
  baik secara meyakinkan di 4096.**

**Speedup terhadap Sequential.** Di laptop 4 core ini, speedup 4 unit pada
uji bergantian turun dari 2,86× (OpenMP, 1024) menjadi 1,29× (OpenMP, 4096),
dan efisiensi dari 71,6% menjadi 32,3%. Hipotesis (belum diuji): algoritma
`i-k-j` tanpa blocking membaca seluruh matriks B (128 MiB pada 4096) untuk
setiap baris A, jauh melebihi kapasitas cache, sehingga bandwidth memori
membatasi kecepatan, bukan jumlah core.

**Hybrid 1×1 lebih lambat daripada Sequential.** Pada benchmark utama,
waktunya 1,04-1,53× waktu Sequential lokal. Fenomena yang mirip muncul pada
program OpenMP Anggota 1 dengan satu thread tanpa MPI sama sekali: pada
data Anggota 1 sendiri OpenMP 1 thread 1,06-1,21× lebih lambat daripada
Sequential, dan pada uji diagnosis di laptop Anggota 3 (1024, 3 pasang
pengukuran bergantian) 1,22-1,54× lebih lambat (uji diagnosis ini tidak
disimpan sebagai CSV). Karena kernel Hybrid, MPI, dan Sequential identik
dalam kodenya, penyebab pastinya belum diketahui. Konsekuensinya, efisiensi
Hybrid 1×1 dan metrik `*_vs_hybrid1x1` harus dibaca dengan hati-hati.

## 11. Batasan Eksperimen

1. **Satu laptop.** Proses MPI berjalan di satu mesin dengan memori bersama,
   bukan cluster. "Komunikasi" MPI di sini adalah penyalinan memori lokal dan
   tidak mencerminkan latensi jaringan.
2. **Hanya 4 core fisik.** Konfigurasi dengan total thread lebih dari 4
   (misalnya 2×4, 4×2, 4×4, dan 8×1 pada panduan tim) tidak diuji, karena akan
   menjadi oversubscription di laptop ini.
3. **Variasi antarsesi besar.** Waktu Sequential 1024 berkisar 0,48-0,94 s dan
   Hybrid 1×4 2048 berkisar 2,94-4,18 s antarsesi. Selisih di bawah variasi ini
   tidak dapat dianggap nyata. Perbandingan antarmetode sebaiknya mengandalkan
   uji bergantian.
4. **Jumlah pengulangan terbatas.** Uji bergantian 4096 hanya 2 putaran, dan
   baseline 4096 di benchmark utama hanya 1 run. Standar proyek (5 pengulangan)
   tetap dipenuhi pada benchmark utama dan pembanding lokal, tetapi bukan pada
   semua baseline.
5. **Validasi.** Hanya checksum; tidak ada perbandingan elemen per elemen.
6. **Perbedaan lingkungan.** Laptop, versi Ubuntu, dan versi Open MPI berbeda
   dengan Anggota 1 dan 2, sehingga angka mereka tidak dibandingkan.
7. **WSL2.** Eksperimen berjalan di lapisan virtualisasi dengan RAM terlihat
   3,7 GiB dari 8 GB fisik; pengaruh lapisan ini terhadap hasil tidak diukur.
8. **Hipotesis.** Penjelasan tentang salinan B per proses dan batasan
   bandwidth memori tidak diverifikasi dengan pengukuran.
