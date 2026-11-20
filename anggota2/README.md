# MatrixX - Implementasi MPI (Anggota 2)

## 1. Deskripsi Proyek
Proyek ini merupakan bagian dari tugas besar UAS Komputasi Paralel dan Terdistribusi (HPC) **MatrixX: Analisis Perbandingan Performa Sequential, MPI, OpenMP, dan Hybrid MPI-OpenMP pada Perkalian Matriks Skala Besar**.

Bagian ini dikerjakan oleh **Anggota 2** dengan fokus pada:
- Implementasi perkalian matriks menggunakan paradigma *Distributed Memory* dengan **MPI (Message Passing Interface)** melalui pustaka **OpenMPI**.
- Evaluasi waktu eksekusi (*Execution Time*), percepatan (*Speedup*), dan efisiensi paralel (*Parallel Efficiency*).
- Validasi akurasi numerik (*Checksum*) terhadap baseline Sequential Anggota 1.
- Visualisasi performa dan analisis mendalam terkait *communication overhead* dan *scalability*.

---

## 2. Struktur Folder
```text
anggota2/
├── src/
│   ├── mpi.cpp            # Source code perkalian matriks OpenMPI C++
│   ├── benchmark.py       # Skrip otomatisasi pengujian 16 konfigurasi
│   └── analysis.py        # Skrip pembuatan grafik dan rekapitulasi metrik
├── results/
│   ├── results_mpi.csv    # Data mentah waktu eksekusi, perulangan, & checksum
│   └── mpi_metrics.csv    # Metrik Speedup dan Parallel Efficiency
├── figures/
│   ├── mpi_execution_time.png   # Grafik waktu eksekusi vs jumlah proses
│   ├── mpi_speedup.png          # Grafik speedup vs baseline Sequential
│   └── mpi_efficiency.png       # Grafik efisiensi paralel (%)
├── docs/
│   └── methodology.md     # Penjelasan metodologi, strategi MPI, dan analisis
├── mpi                    # File biner hasil kompilasi
└── README.md              # Panduan instruksi dan dokumentasi
```

---

## 3. Lingkungan Pengujian
- **Operating System:** Windows 11 melalui Ubuntu 24.04 LTS (WSL2)
- **Bahasa Pemrograman:** C++ (Standar C++17)
- **Compiler:** `g++` (Ubuntu 13.3.0) & `mpicxx`
- **Pustaka MPI:** OpenMPI 4.1.6
- **Python & Visualisasi:** Python 3, Matplotlib, Pandas

---

## 4. Cara Kompilasi dan Eksekusi

### A. Kompilasi Program MPI
Jalankan perintah kompilasi dengan optimasi `-O3` dan instruksi arsitektur lokal:
```bash
mpicxx -O3 -march=native -std=c++17 anggota2/src/mpi.cpp -o anggota2/mpi
```

### B. Menjalankan Pengujian Manual
Format perintah:
```bash
mpirun --allow-run-as-root -np <JUMLAH_PROSES> ./anggota2/mpi <UKURAN_MATRIKS> <PENGULANGAN>
```
Contoh (Ukuran 1024x1024 dengan 4 proses, 5 pengulangan):
```bash
mpirun --allow-run-as-root -np 4 ./anggota2/mpi 1024 5
```

### C. Menjalankan Benchmark Penuh Otomatis
Untuk menguji seluruh konfigurasi (ukuran 512, 1024, 2048, 4096 pada 1, 2, 4, 8 proses):
```bash
python3 anggota2/src/benchmark.py
```

### D. Menghasilkan Grafik Analisis
Untuk membuat seluruh visualisasi grafik di folder `figures/`:
```bash
python3 anggota2/src/analysis.py
```

---

## 5. Ringkasan Hasil Eksperimen

Baseline Sequential (Anggota 1):
- $512 \times 512$: $0.033638$ detik
- $1024 \times 1024$: $0.452295$ detik
- $2048 \times 2048$: $4.996365$ detik
- $4096 \times 4096$: $37.596400$ detik

### Hasil Pengujian MPI (Rata-rata 5 Pengulangan):

| Ukuran Matriks | Proses ($P$) | Waktu Eksekusi (s) | Speedup ($S$) | Efficiency ($E$) | Checksum (Validasi) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **512 x 512** | 1 | 0.155967 | 0.22x | 21.57% | 32863525.795196988 |
| | 2 | 0.093979 | 0.36x | 17.90% | 32863525.795196988 |
| | 4 | 0.111898 | 0.30x | 7.52% | 32863525.795196988 |
| | 8 | 0.102261 | 0.33x | 4.11% | 32863525.795196988 |
| **1024 x 1024** | 1 | 0.999485 | 0.45x | 45.25% | 263006890.099257082 |
| | 2 | 0.567473 | 0.80x | 39.85% | 263006890.099257082 |
| | 4 | 0.442353 | 1.02x | 25.56% | 263006890.099257082 |
| | 8 | 0.300025 | 1.51x | 18.84% | 263006890.099257082 |
| **2048 x 2048** | 1 | 6.464330 | 0.77x | 77.29% | 2104662477.597485304 |
| | 2 | 3.635910 | 1.37x | 68.71% | 2104662477.597485304 |
| | 4 | 2.576086 | 1.94x | 48.49% | 2104662477.597485304 |
| | 8 | 2.113662 | 2.36x | 29.55% | 2104662477.597485304 |
| **4096 x 4096** | 1 | 57.466853 | 0.65x | 65.42% | 16838019066.392269135 |
| | 2 | 33.197549 | 1.13x | 56.63% | 16838019066.392269135 |
| | 4 | 20.693464 | 1.82x | 45.42% | 16838019066.392269135 |
| | 8 | 17.586926 | 2.14x | 26.72% | 16838019066.392269135 |

---

## 6. Analisis Utama
1. **Validasi Numerik 100%:** Seluruh Checksum pada setiap ukuran matriks identik hingga digit desimal terakhir dengan hasil Sequential Anggota 1, membuktikan kebenaran matematis algoritma perkalian dan dekomposisi data.
2. **Pengaruh Skala Komputasi vs Komunikasi:** 
   - Pada ukuran kecil ($512 \times 512$), waktu komunikasi (`MPI_Scatterv`, `MPI_Bcast`, `MPI_Gatherv`) mendominasi dibandingkan komputasi yang sangat singkat, sehingga speedup belum terlihat signifikan.
   - Pada ukuran besar ($1024$, $2048$, $4096$), beban komputasi melonjak ($O(N^3)$), sehingga paralelisasi MPI memberikan peningkatan performa yang sangat nyata, dengan percepatan hingga **2.36x**.
3. **Keterbatasan Single Node (Laptop):** Pengujian MPI dilakukan pada satu laptop fisik, di mana seluruh proses berbagi memori fisik, *bus* memori, dan *L3 cache* yang sama (*resource contention*).
