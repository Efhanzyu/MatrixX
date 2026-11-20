# Metodologi Eksperimen MPI - Anggota 2

## 1. Ruang Lingkup dan Tanggung Jawab
Dokumen ini menguraikan metodologi komputasi paralel menggunakan **MPI (Message Passing Interface)** pada proyek **MatrixX**.
Sebagai **Anggota 2**, ruang lingkup pengujian berfokus pada:
1. Implementasi algoritma perkalian matriks menggunakan arsitektur *distributed memory* berbasis proses MPI.
2. Strategi partisi data (*domain decomposition*) dan komunikasi kolektif antar-proses.
3. Pengujian skala performa (*strong scaling*) pada ukuran matriks 512, 1024, 2048, dan 4096 dengan 1, 2, 4, dan 8 proses.
4. Analisis komparatif terhadap baseline *Sequential* yang telah diselesaikan oleh Anggota 1.

---

## 2. Model Komputasi dan Algoritma

### 2.1. Spesifikasi Operasi
Diberikan dua matriks kuadrat $A, B \in \mathbb{R}^{N \times N}$, perkalian matriks menghasilkan matriks $C \in \mathbb{R}^{N \times N}$ dengan elemen:
$$C[i][j] = \sum_{k=0}^{N-1} A[i][k] \cdot B[k][j]$$

### 2.2. Pola Inisialisasi Data (Standar Proyek)
Untuk memastikan konsistensi numerik antar seluruh anggota tim:
```cpp
A[i][j] = ((i + j) % 100) * 0.01;
B[i][j] = ((i * 3 + j) % 100) * 0.01;
```
Tipe data yang digunakan adalah `double` (presisi ganda, 64-bit IEEE 754) yang disimpan dalam struktur kontigu (vektor 1D) untuk memaksimalkan *spatial locality* memori.

### 2.3. Optimalisasi Loop Memory Access
Algoritma perkalian lokal di setiap proses worker menerapkan permutasi loop **$i-k-j$**:
```cpp
for (int i = 0; i < my_rows; i++) {
    for (int k = 0; k < N; k++) {
        double temp = A_local[i * N + k];
        for (int j = 0; j < N; j++) {
            C_local[i * N + j] += temp * B[k * N + j];
        }
    }
}
```
Loop dalam ($j$) mengakses baris matriks $C$ dan baris matriks $B$ secara sekuensial (*stride-1 access*), meminimalisir *cache misses* pada arsitektur CPU modern.

---

## 3. Strategi Dekomposisi dan Komunikasi MPI

Pada komputasi paralel berbasis MPI, setiap proses memiliki ruang memori independen (*private memory space*).

```
   Rank 0 (Master)
  ┌─────────────────────────────────────────────────────────────┐
  │ Inisialisasi Matriks A dan B                                │
  └──────────────┬───────────────────────────────┬──────────────┘
                 │                               │
       MPI_Scatterv (Baris A)          MPI_Bcast (Matriks B Penuh)
                 │                               │
        ┌────────┴────────┬──────────────┬───────┴────────┐
        ▼                 ▼              ▼                ▼
     Rank 0            Rank 1         Rank 2           Rank 3
  ┌───────────┐     ┌───────────┐  ┌───────────┐    ┌───────────┐
  │ Hitung C0 │     │ Hitung C1 │  │ Hitung C2 │    │ Hitung C3 │
  └─────┬─────┘     └─────┬─────┘  └─────┬─────┘    └─────┬─────┘
        │                 │              │                │
        └────────┬────────┴──────────────┴────────────────┘
                 │
        MPI_Gatherv (Satukan Sub-Matriks C)
                 ▼
   Rank 0: Checksum & Pengukuran Waktu
```

1. **Dekomposisi Baris Matriks A (`MPI_Scatterv`):**
   - Matriks $A$ dibagi secara horizontal (blok baris) ke $P$ proses.
   - Menggunakan `MPI_Scatterv` dengan perhitungan `sendcounts` dan `displs` fleksibel sehingga tetap valid dan stabil meski ukuran matriks $N$ tidak habis dibagi oleh jumlah proses $P$.
2. **Replikasi Matriks B (`MPI_Bcast`):**
   - Karena setiap proses membutuhkan seluruh kolom matriks $B$ untuk menghitung baris hasil $C$, matriks $B$ lengkap didistribusikan ke seluruh proses melalui komunikasi kolektif satu-ke-semua (`MPI_Bcast`).
3. **Pengumpulan Matriks C (`MPI_Gatherv`):**
   - Setiap worker menghitung sub-matriks $C_{\text{local}} \in \mathbb{R}^{\text{my\_rows} \times N}$.
   - Seluruh sub-matriks hasil disatukan kembali ke memori Rank 0 menggunakan `MPI_Gatherv`.

---

## 4. Metodologi Pengukuran dan Validasi

### 4.1. Pengukuran Waktu (*Timing*)
- Waktu eksekusi diukur menggunakan fungsi presisi tinggi `MPI_Wtime()`.
- Pemanggilan `MPI_Barrier(MPI_COMM_WORLD)` dilakukan tepat sebelum pencatatan waktu dimulai dan setelah seluruh proses selesai, memastikan waktu yang tercatat mencakup keseluruhan siklus komunikasi (*Scatter/Bcast*) dan komputasi paralel.
- Setiap pengujian diulang sebanyak **5 kali** (*5 repeats*). Nilai yang dianalisis adalah rata-rata (*Average Time*) dan nilai minimum (*Minimum Time*).

### 4.2. Validasi Kebenaran (*Checksum*)
Validasi dilakukan dengan menghitung jumlah akumulatif seluruh elemen pada matriks hasil $C$:
$$\text{Checksum} = \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} C[i][j]$$

Hasil verifikasi:
- $N = 512$: `32863525.795196988` (Identik 100% dengan Sequential)
- $N = 1024$: `263006890.099257082` (Identik 100% dengan Sequential)
- $N = 2048$: `2104662477.597485304` (Identik 100% dengan Sequential)
- $N = 4096$: `16838019066.392269135` (Identik 100% dengan Sequential)

---

## 5. Metrik Evaluasi

1. **Speedup ($S$):**
   $$S = \frac{T_{\text{seq}}}{T_{\text{mpi}}}$$
   Mengukur rasio percepatan eksekusi MPI relatif terhadap eksekusi sekuensial Anggota 1.

2. **Parallel Efficiency ($E$):**
   $$E = \left(\frac{S}{P}\right) \times 100\%$$
   Mengukur efektivitas pemanfaatan setiap proses yang dialokasikan ($P \in \{1, 2, 4, 8\}$).

---

## 6. Analisis Hasil Eksperimen

1. **Trade-off Komunikasi vs Komputasi:**
   - Pada ukuran $N=512$, beban komputasi per proses sangat kecil ($O(N^3) \approx 1.34 \times 10^8$ FLOPs). Biaya *overhead* komunikasi antar-proses (alokasi buffer, serialisasi memori, dan penyalinan data via IPC kernel) lebih dominan dibandingkan waktu komputasinya.
   - Pada ukuran $N=2048$ dan $N=4096$, beban komputasi melonjak drastis hingga miliaran operasi floating-point. Pada titik ini, rasio komputasi terhadap komunikasi menjadi sangat tinggi, sehingga pembagian tugas ke banyak proses menghasilkan penurunan waktu eksekusi yang tajam (waktu turun dari 57.46 detik pada 1 proses menjadi 17.58 detik pada 8 proses).

2. **Karakteristik Skalabilitas pada Single-Node:**
   - Eksekusi MPI pada laptop tunggal menggunakan mekanisme *Inter-Process Communication* (IPC / *shared-memory transport*) di bawah sistem operasi.
   - Penambahan proses dari 4 ke 8 proses pada ukuran 4096 menghasilkan peningkatan performa tambahan (20.69s $\rightarrow$ 17.58s), meskipun kurva efisiensi melandai karena keterbatasan *memory bandwidth* pada arsitektur memori terintegrasi laptop.
