# MatrixX - Panduan Deployment Cluster & Monitoring Dashboard

Dokumentasi ini mencakup implementasi **Bonus (+5%)** pada Tugas Besar UAS Komputasi Paralel dan Terdistribusi:
1. **Containerized HPC Cluster (Docker Compose)**
2. **Kubernetes Orchestration Manifests (`k8s/`)**
3. **Web Real-time Monitoring Dashboard (`dashboard/`)**

---

## 1. Arsitektur Topologi Cluster

Cluster dirancang dengan konsep **Master-Worker Multi-Node**:
- **Master Node (`master-node`)**: IP `172.20.0.2` — Bertindak sebagai koordinator task scheduler dan eksekutor utama `mpirun`.
- **Worker 1 (`worker-1`)**: IP `172.20.0.3` — Node komputasi paralel MPI Rank 1.
- **Worker 2 (`worker-2`)**: IP `172.20.0.4` — Node komputasi paralel MPI Rank 2.
- **Dashboard (`dashboard`)**: IP `172.20.0.5` — Web monitoring service pada port `5000`.
- **Shared Storage**: Volume bersama yang di-mount pada seluruh node di direktori `/workspace`.

---

## 2. Cara Menjalankan Monitoring Dashboard secara Lokal (Native Python)

Jika Docker belum terpasang di host, dashboard dapat langsung dijalankan menggunakan Python:

```bash
cd dashboard
python app.py
```

Buka browser Anda di: **`http://localhost:5000`**

### Fitur Dashboard:
- 🖥️ **Status Live Node**: Menampilkan status online, IP, peran, dan utilisasi Master & Worker nodes.
- 📊 **Metrik Real-time**: CPU Load, Memory Load, dan grafik perbandingan performa.
- ⚡ **Interactive Benchmark Runner**: Pemicu pengujian matriks $512 \times 512$ hingga $4096 \times 4096$ dengan stream log interaktif.
- 📈 **Grafik Interaktif (Chart.js)**: Visualisasi waktu eksekusi logaritmik dan kurva Speedup vs Amdahl's Law.

---

## 3. Deployment Berbasis Docker Compose

Jika Docker Desktop sudah terpasang:

```bash
cd deploy
docker compose up -d --build
```

Memeriksa status container cluster:
```bash
docker compose ps
```

Masuk ke dalam Master Node untuk eksekusi MPI manual:
```bash
docker exec -it matrixx-master bash
mpicxx -O3 -std=c++17 anggota2/src/mpi.cpp -o anggota2/mpi
mpirun --host master-node,worker-1,worker-2 -np 4 ./anggota2/mpi 2048 5
```

Menghentikan cluster:
```bash
docker compose down
```

---

## 4. Deployment Berbasis Kubernetes (`k8s/`)

Untuk deployment pada lingkungan Kubernetes (Minikube / K3s / MicroK8s):

```bash
kubectl apply -f k8s/01-cluster-setup.yaml
kubectl apply -f k8s/02-master-workers.yaml
kubectl apply -f k8s/03-dashboard.yaml
```

Verifikasi Pods dan Service:
```bash
kubectl get pods -n matrixx-hpc
kubectl get svc -n matrixx-hpc
```
Akses dashboard pada port NodePort: `http://<Node-IP>:30500`.
