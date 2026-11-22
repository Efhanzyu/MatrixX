# MatrixX: High-Performance Computing (HPC) for Matrix Computation

Proyek Tugas Besar Ujian Akhir Semester (UAS) **Komputasi Paralel dan Terdistribusi (3 SKS)**  
Program Studi Informatika, Fakultas Teknik, Universitas Siliwangi  
Dosen Pengampu: **Ir. Randi Rizal, Ph.D.**

---

## 1. Deskripsi Proyek

**MatrixX** adalah sistem komputasi paralel dan terdistribusi berskala besar untuk analisis perbandingan perkalian matriks hingga ukuran **$4096 \times 4096$** (16.7 juta elemen, ~134 MB data double precision). Proyek ini membandingkan empat paradigma komputasi:
1. **Sequential (Baseline)**: Eksekusi berurutan single-thread pada CPU host.
2. **OpenMP (Shared Memory)**: Komputasi paralel multi-threading berbasis shared memory.
3. **MPI (Distributed Memory)**: Distribusi beban kerja antar proses terdistribusi via OpenMPI.
4. **Hybrid (MPI + OpenMP)**: Model hierarkis tercanggih yang menggabungkan komunikasi inter-node via MPI dan paralelisme intra-node via OpenMP multi-threading.
5. **Monitoring & Container Deployment (Bonus +5%)**: Klaster virtual berbasis Docker Compose, orkestrasi Kubernetes, serta Real-time Web Monitoring Dashboard.

---

## 2. Anggota Tim & Pembagian Tugas

| No | Peran | Tugas & Modul yang Dikerjakan | Folder |
|:---:|:---|:---|:---:|
| 1 | **Anggota 1** *(Lead / Reviewer)* | Implementasi Baseline Sequential & OpenMP, Benchmark Awal & Validasi 10x, Metodologi, dan Koordinasi Repositori. | [`anggota1/`](./anggota1/) |
| 2 | **Anggota 2** | Implementasi Distributed Memory MPI, Benchmark Otomasi 16 Konfigurasi, Analisis Communication Overhead. | [`anggota2/`](./anggota2/) |
| 3 | **Anggota 3** | Implementasi Hybrid MPI+OpenMP, Skrip Benchmark Komparasi Lintas Metode, dan Visualisasi Terpadu. | [`anggota3/`](./anggota3/) |
| - | **Sistem & Integrasi** | Container Cluster Docker, Kubernetes Manifests, dan Web Monitoring Dashboard. | [`deploy/`](./deploy/) & [`dashboard/`](./dashboard/) |

---

## 3. Arsitektur Sistem & Topologi Klaster (Poin B Naskah Soal)

Sistem dirancang memenuhi seluruh spesifikasi arsitektur komputasi terdistribusi:
1. **Master / Coordinator Node**: Bertanggung jawab atas inisialisasi matriks, task scheduling, pembagian partisi baris (*domain decomposition*), dan pengumpulan hasil (*MPI_Gather*).
2. **Worker Nodes (Worker 1 & Worker 2)**: Node komputasi paralel yang mengeksekusi perkalian blok matriks menggunakan OpenMP multi-threads.
3. **Communication Layer**: Protokol OpenMPI untuk transfer data inter-node (`MPI_Bcast` untuk matriks B dan `MPI_Scatterv`/`MPI_Gatherv` untuk matriks A dan C).
4. **Shared / Distributed Storage**: Volume bersama (`/workspace`) yang di-mount ke seluruh container.
5. **Monitoring Module**: Web-based real-time dashboard yang melacak status live node, latency ping, beban CPU, dan memori.

---

## 4. Struktur Folder Repositori

```text
MatrixX/
├── anggota1/                # Modul Sequential & OpenMP (Anggota 1)
│   ├── src/                 # Source code C++ & analisis Python
│   ├── results/             # CSV metrik benchmark & validasi
│   ├── figures/             # Grafik waktu, speedup, & efisiensi
│   └── docs/                # Dokumentasi metodologi
├── anggota2/                # Modul MPI Terdistribusi (Anggota 2)
│   ├── src/                 # Source code OpenMPI C++ & otomatisasi
│   ├── results/             # CSV metrik MPI
│   ├── figures/             # Grafik scaling MPI
│   └── docs/                # Analisis communication overhead
├── anggota3/                # Modul Hybrid MPI + OpenMP (Anggota 3)
│   ├── src/                 # Source code Hybrid C++ & analysis_compare.py
│   ├── results/             # CSV rekapitulasi komparasi seluruh metode
│   ├── figures/             # Grafik komparasi langsung
│   └── docs/                # Analisis perbandingan komputasi
├── dashboard/               # Real-time Web Monitoring Dashboard (Bonus +5%)
│   ├── app.py               # Flask REST API & Web Server
│   ├── templates/index.html # Modern Dark-mode Dashboard UI
│   └── requirements.txt     # Dependensi Python
├── deploy/                  # Deployment Docker & Kubernetes (Bonus +5%)
│   ├── Dockerfile.cluster   # Image cluster HPC Ubuntu 24.04 + OpenMPI
│   ├── docker-compose.yml   # Multi-node cluster (Master + 2 Workers + Dashboard)
│   ├── k8s/                 # Kubernetes Manifests (Pods, Services, PVC)
│   └── README.md            # Panduan teknis deployment
└── README.md                # Dokumentasi utama proyek
```

---

## 5. Ringkasan Hasil Eksperimen (Matriks 4096×4096)

Pengujian dilakukan pada ukuran matriks maksimum $4096 \times 4096$ dengan 5 kali pengulangan:

| Paradigma Komputasi | Konfigurasi Core/Thread | Rata-rata Waktu (s) | Speedup ($S$) | Efisiensi ($E$) | Validasi Checksum |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Sequential (Baseline)** | 1 Core | 53.323 s | 1.00x | 100.0% | Valid |
| **OpenMP** | 4 Threads | 36.519 s | **1.46x** | 36.5% | Valid |
| **MPI Terdistribusi** | 4 Processes | 39.918 s | **1.34x** | 33.4% | Valid |
| **Hybrid (MPI+OpenMP)** | 2 Proc × 2 Threads | 39.715 s | **1.34x** | 33.6% | Valid |

> **Catatan Analisis**: Pada matriks yang lebih kecil ($512$ dan $1024$), **Hybrid MPI+OpenMP** menghasilkan percepatan paling drastis hingga **3.51x** berkat rendahnya contention memori dibanding OpenMP murni dan efisiensi komunikasi scatter-gather dibanding MPI murni.

---

## 6. Cara Menjalankan Sistem

### A. Menjalankan Web Monitoring Dashboard (Lokal)
```bash
cd dashboard
python app.py
```
Akses di browser: `http://localhost:5000`

### B. Menjalankan Cluster Docker
```bash
cd deploy
docker compose up -d --build
```

### C. Menjalankan Benchmark Mandiri di WSL / Linux
```bash
# Kompilasi & Jalankan OpenMP (Anggota 1)
g++ -O3 -march=native -std=c++17 -fopenmp anggota1/src/openmp.cpp -o anggota1/openmp
./anggota1/openmp 2048 5

# Kompilasi & Jalankan MPI (Anggota 2)
mpicxx -O3 -march=native -std=c++17 anggota2/src/mpi.cpp -o anggota2/mpi
mpirun -np 4 ./anggota2/mpi 2048 5

# Kompilasi & Jalankan Hybrid (Anggota 3)
mpicxx -O3 -march=native -std=c++17 -fopenmp anggota3/src/hybrid.cpp -o anggota3/hybrid
mpirun -np 2 ./anggota3/hybrid 2048 5 2
```
