
# MatrixX - Sequential dan OpenMP

## 1. Deskripsi

MatrixX merupakan proyek analisis perbandingan performa
perkalian matriks menggunakan pendekatan komputasi
Sequential dan paralel OpenMP.

Implementasi Anggota 1 mencakup:
- Sequential Matrix Multiplication
- OpenMP Matrix Multiplication
- Benchmark berbagai ukuran matriks
- Pengukuran speedup dan parallel efficiency
- Visualisasi hasil eksperimen

## 2. Lingkungan Pengembangan

- Processor: Intel Core i5-12450H
- CPU tersedia di WSL2: 12 logical CPU
- RAM tersedia di WSL2: 7.6 GiB
- Operating System: Ubuntu melalui WSL2
- Programming Language: C++
- Compiler: G++ 15.2.0
- Python: 3.14.4
- Matplotlib: 3.10.7
- C++ Standard: C++17

## 3. Struktur Folder

```text
anggota1/
├── src/
│   ├── sequential.cpp
│   ├── openmp.cpp
│   ├── analysis.py
│   └── validation_analysis.py
├── results/
│   ├── results_sequential.csv
│   ├── results_openmp.csv
│   ├── results_validation.csv
│   ├── performance_metrics.csv
│   └── validation_metrics.csv
├── figures/
│   ├── execution_time.png
│   ├── strong_scaling.png
│   ├── speedup.png
│   ├── parallel_efficiency.png
│   ├── validation_execution_time.png
│   ├── validation_speedup.png
│   └── validation_efficiency.png
├── docs/
└── README.md
```

## 4. Kompilasi

### Sequential

```bash
g++ -O3 -march=native -std=c++17 \
src/sequential.cpp -o sequential
```

### OpenMP

```bash
g++ -O3 -march=native -std=c++17 -fopenmp \
src/openmp.cpp -o openmp
```

## 5. Menjalankan Program

### Sequential

```bash
./sequential 1024 5
```

### OpenMP

```bash
OMP_NUM_THREADS=4 ./openmp 1024 5
```

Parameter pertama adalah ukuran matriks.
Parameter kedua adalah jumlah pengulangan.

## 6. Analisis Benchmark

Menjalankan analisis benchmark awal:

```bash
python3 src/analysis.py
```

Menjalankan analisis validasi:

```bash
python3 src/validation_analysis.py
```

## 7. Ukuran Matriks dan Konfigurasi

Ukuran matriks:
- 512 x 512
- 1024 x 1024
- 2048 x 2048
- 4096 x 4096

Konfigurasi thread OpenMP:
- 1 thread
- 2 threads
- 4 threads
- 8 threads

## 8. Parameter Evaluasi

Speedup:

S = T_sequential / T_parallel

Parallel Efficiency:

E = S / P

Keterangan:
- S = Speedup
- E = Parallel Efficiency
- P = Jumlah thread
- T = Waktu eksekusi

## 9. Catatan

Benchmark awal menggunakan 5 pengulangan.
Validasi tambahan menggunakan 10 pengulangan.

Data validasi disimpan terpisah dari data benchmark awal.
