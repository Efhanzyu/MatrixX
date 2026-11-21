# MatrixX - Implementasi Hybrid MPI-OpenMP (Anggota 3)

## 1. Deskripsi

Folder ini berisi implementasi perkalian matriks **Hybrid MPI-OpenMP** untuk
proyek MatrixX: *Analisis Perbandingan Performa Sequential, MPI, OpenMP, dan
Hybrid MPI-OpenMP pada Perkalian Matriks Skala Besar*.

MPI membagi baris matriks A antarproses, dan OpenMP membagi baris lokal
tersebut ke beberapa thread di dalam setiap proses. Algoritma, tipe data
(`double`), pola data, dan urutan loop (`i-k-j`) mengikuti standar Anggota 1
dan 2. Source code Anggota 1 dan 2 tidak diubah.

Penjelasan lengkap metodologi, hasil, dan analisis ada di
[`docs/methodology.md`](docs/methodology.md).

## 2. Struktur Folder

```text
anggota3/
├── README.md
├── src/
│   ├── hybrid.cpp                  # implementasi Hybrid MPI-OpenMP
│   ├── benchmark.py                # benchmark utama Hybrid + baseline Sequential lokal
│   ├── benchmark_compare.py        # OpenMP dan MPI (Anggota 1 dan 2) di laptop ini
│   ├── benchmark_interleaved.py    # uji bergantian lintas metode
│   ├── analysis.py                 # grafik Hybrid
│   └── analysis_compare.py         # grafik perbandingan antarmetode
├── results/
│   ├── results_hybrid.csv          # hasil benchmark utama Hybrid
│   ├── hybrid_metrics.csv          # speedup, efisiensi, overhead
│   ├── results_sequential_local.csv# baseline Sequential di laptop ini
│   ├── comparison_local.csv        # OpenMP dan MPI (1, 2, 4 unit)
│   ├── interleaved_comparison.csv  # uji bergantian 1024 dan 2048 (data mentah)
│   ├── interleaved_summary.csv     # ringkasan uji bergantian 1024 dan 2048
│   ├── interleaved_4096/           # uji bergantian 4096
│   └── system_info.txt             # spesifikasi dan versi perangkat lunak
├── figures/
│   ├── hybrid_execution_time.png
│   ├── hybrid_speedup.png
│   ├── hybrid_efficiency.png
│   ├── hybrid_time_breakdown.png
│   ├── compare_execution_time.png
│   └── compare_speedup_efficiency.png
└── docs/
    └── methodology.md
```

## 3. Lingkungan Pengujian

| Komponen | Spesifikasi |
|---|---|
| Laptop | Acer Aspire AL14-37P |
| CPU | Intel N150, 4 core, 1 thread per core (4 logical CPU) |
| RAM | 8192 MB fisik; 3,7 GiB terlihat di WSL2; swap 1,0 GiB |
| OS | Windows 11 Home 26H2 dengan WSL2, Ubuntu 26.04.1 LTS |
| Compiler | g++ 15.2.0, C++17 |
| MPI | Open MPI 5.0.10 |
| Python | 3.14.4, Matplotlib |

Versi lengkap tersimpan di `results/system_info.txt`. Laptop, versi Ubuntu, dan
versi Open MPI berbeda dengan Anggota 1 dan 2, sehingga **angka waktu
antaranggota tidak boleh dibandingkan langsung**.

## 4. Cara Kompilasi dan Eksekusi

### A. Dependensi (Ubuntu / WSL2)

```bash
sudo apt update
sudo apt install -y g++ make git openmpi-bin libopenmpi-dev python3 python3-matplotlib
```

### B. Kompilasi

Skrip benchmark memerlukan keempat program. Jalankan dari root repository:

```bash
g++ -O3 -march=native -std=c++17 anggota1/src/sequential.cpp -o anggota1/sequential
g++ -O3 -march=native -std=c++17 -fopenmp anggota1/src/openmp.cpp -o anggota1/openmp
mpicxx -O3 -march=native -std=c++17 anggota2/src/mpi.cpp -o anggota2/mpi
mpicxx -O3 -march=native -std=c++17 -fopenmp anggota3/src/hybrid.cpp -o anggota3/hybrid
```

Binary otomatis diabaikan Git (`.gitignore`).

### C. Menjalankan manual

Argumen: `<ukuran matriks> <jumlah pengulangan>`. Jumlah thread per proses
diatur lewat `OMP_NUM_THREADS`, jumlah proses lewat `-np`:

```bash
# MPI 2 proses x OpenMP 2 thread, matriks 1024, 5 pengulangan
OMP_NUM_THREADS=2 mpirun --bind-to none -np 2 ./anggota3/hybrid 1024 5
```

Opsi `--bind-to none` mencegah Open MPI mengunci semua thread sebuah proses
ke satu core. Jangan memakai total thread (proses x thread) di atas jumlah core
fisik kecuali sengaja menguji oversubscription.

Contoh keluaran:

```text
Method: Hybrid
Matrix Size: 1024 x 1024
Processes: 2
Threads per Process: 2
Total Threads: 4
Average Execution Time: ... seconds
Minimum Execution Time: ... seconds
Average Compute Time: ... seconds
Checksum: 263006890.099257082
```

### D. Menjalankan benchmark penuh

Urutan penting, karena skrip berikutnya memakai baseline dari skrip pertama.
Charger terpasang, mode daya *Best performance*, tidak ada aplikasi berat lain:

```bash
# 1. Benchmark utama Hybrid + baseline Sequential lokal (sekitar 35 menit)
python3 anggota3/src/benchmark.py 2>&1 | tee ~/benchmark_hybrid.log

# 2. Pembanding lokal OpenMP dan MPI (sekitar 30 menit)
python3 anggota3/src/benchmark_compare.py 2>&1 | tee ~/benchmark_compare.log

# 3. Uji bergantian ukuran 1024 dan 2048 (perkiraan 5-8 menit)
python3 anggota3/src/benchmark_interleaved.py 2>&1 | tee ~/benchmark_interleaved.log

# 4. Uji bergantian ukuran 4096, 2 putaran (perkiraan sekitar 20 menit)
python3 anggota3/src/benchmark_interleaved.py --sizes 4096 --rounds 2 \
    --out-dir anggota3/results/interleaved_4096 2>&1 | tee ~/benchmark_interleaved_4096.log
```

Perkiraan waktu langkah 1 dan 2 berasal dari log eksekusi pada laptop ini;
langkah 3 dan 4 adalah perkiraan. Menjalankan ulang akan menimpa CSV di
`results/`.

### E. Menghasilkan grafik

```bash
python3 anggota3/src/analysis.py
python3 anggota3/src/analysis_compare.py
```

## 5. Ringkasan Hasil Eksperimen

Seluruh konfigurasi pada ketiga eksperimen berstatus checksum cocok dengan
Sequential lokal (toleransi relatif 1e-12). Tabel di bawah adalah **uji
bergantian** (4 unit pemrosesan), yaitu perbandingan paling andal karena
semua metode diukur selang-seling dalam satu sesi. Nilai adalah median
antarputaran; speedup terhadap Sequential pada sesi yang sama.

| Ukuran | OpenMP 4 thread | Hybrid 1x4 | Hybrid 2x2 | MPI 4 proses |
|---|---|---|---|---|
| 1024 (3 putaran) | 0,1998 s (2,86x) | 0,2060 s (2,78x) | 0,3262 s (1,75x) | 0,4843 s (1,18x) |
| 2048 (3 putaran) | 2,8808 s (1,92x) | 2,9386 s (1,88x) | 3,7474 s (1,47x) | 4,1844 s (1,32x) |
| 4096 (2 putaran) | 36,6387 s (1,29x) | 39,1707 s (1,21x) | 39,6318 s (1,20x) | 41,2698 s (1,15x) |

Baseline Sequential pada uji bergantian: 0,5725 s (1024), 5,5260 s (2048),
dan 47,3992 s (4096). Tabel lengkap benchmark utama (24 konfigurasi Hybrid)
ada di `docs/methodology.md`, bagian 9.1.

## 6. Analisis Utama

- **Kombinasi terbaik di laptop ini adalah Hybrid 1x4** (satu proses, empat
  thread). Lebih banyak proses MPI dengan total thread yang sama membuat waktu
  lebih lama, dan pengaruhnya mengecil seiring ukuran matriks.
- **Hybrid 1x4 dan OpenMP 4 thread setara** (selisih 2-7% dengan rentang yang
  tumpang tindih). MPI 4 proses paling lambat di 1024 dan 2048.
- **Pada 4096, tidak ada metode yang terbukti lebih baik**: selisih tercepat
  dan terlambat hanya 12,6% dan rentang antarputaran saling tumpang tindih
  (hanya 2 putaran).
- **Overhead non-komputasi** (komunikasi, sinkronisasi, ketidakseimbangan beban)
  hanya 0,18-6,14% dari waktu total, dan tidak lebih dari 0,94% pada 2048 dan
  4096. Selisih antarkombinasi pada ukuran besar berasal dari komputasi.
- **Speedup 4 unit hanya 1,15-1,29x pada 4096** (efisiensi 29-32%). Penjelasan
  yang diajukan, yaitu keterbatasan bandwidth memori dan salinan B per proses,
  adalah hipotesis yang belum diuji.
- **Hybrid 1x1 lebih lambat daripada Sequential** (1,04-1,53x pada benchmark
  utama). Pola serupa muncul pada OpenMP Anggota 1 dengan satu thread, jadi
  tampaknya bukan bug di `hybrid.cpp`, tetapi penyebab pastinya belum
  diketahui.

## 7. Catatan dan Keterbatasan

1. Pengujian pada **satu laptop 4 core**: proses MPI berbagi memori, bukan
   cluster fisik, sehingga komunikasinya tidak mencerminkan latensi jaringan.
2. Konfigurasi dengan total thread di atas 4 (misalnya 2x4, 4x2, 4x4, dan 8x1
   pada panduan tim) tidak diuji karena akan menjadi oversubscription.
3. Waktu di laptop ini bervariasi besar antarsesi (Sequential 1024 berkisar
   0,48-0,94 s). Selisih kecil antarmetode jangan ditafsirkan sebagai nyata.
4. Baseline Sequential 4096 pada benchmark utama hanya 1 run dan 12,5% lebih
   lambat daripada pada uji bergantian, sehingga speedup 4096 pada benchmark
   utama cenderung terlalu tinggi. Gunakan tabel uji bergantian.
5. Validasi memakai checksum saja, tanpa perbandingan elemen per elemen.
6. Efisiensi di atas 100% pada beberapa konfigurasi 1x2 adalah artefak
   referensi satu thread yang lambat, bukan speedup superlinear yang nyata.

### Pesan yang bisa diabaikan di WSL2

- `mpirun --version` menampilkan pesan *help file* yang tidak ditemukan. Itu
  masalah paket Open MPI 5.0.10 di Ubuntu ini, program tetap berjalan.
- `Failed to connect to system scope bus` saat `apt install` tidak
  memengaruhi instalasi paket.
