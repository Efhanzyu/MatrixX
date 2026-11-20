import subprocess
import csv
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANGGOTA1_RESULTS = os.path.join(ROOT_DIR, "anggota1", "results", "results_sequential.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Baca baseline Sequential dari Anggota 1
seq_baseline = {}
if os.path.exists(ANGGOTA1_RESULTS):
    with open(ANGGOTA1_RESULTS, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_baseline[int(row["matrix_size"])] = float(row["average_time"])
else:
    print(f"Warning: File {ANGGOTA1_RESULTS} tidak ditemukan. Menggunakan fallback standar.")
    seq_baseline = {
        512: 0.033638255,
        1024: 0.452294608,
        2048: 4.996364580,
        4096: 37.596399516
    }

matrix_sizes = [512, 1024, 2048, 4096]
process_counts = [1, 2, 4, 8]
repeats = 5

results_raw = []
metrics = []

total_runs = len(matrix_sizes) * len(process_counts)
current_run = 0

print("=" * 65)
print("       BENCHMARK PARALEL MPI MATRIXX (ANGGOTA 2)")
print("=" * 65)

for size in matrix_sizes:
    for p in process_counts:
        current_run += 1
        print(f"[{current_run}/{total_runs}] Menguji Ukuran {size}x{size} dengan {p} Proses MPI... ", end="", flush=True)

        cmd = [
            "mpirun",
            "--allow-run-as-root",
            "--oversubscribe",
            "-np", str(p),
            os.path.join(BASE_DIR, "mpi"),
            str(size),
            str(repeats)
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            output = res.stdout

            avg_time = None
            min_time = None
            checksum = None

            for line in output.splitlines():
                if "Average Execution Time:" in line:
                    avg_time = float(line.split(":")[1].replace("seconds", "").strip())
                elif "Minimum Execution Time:" in line:
                    min_time = float(line.split(":")[1].replace("seconds", "").strip())
                elif "Checksum:" in line:
                    checksum = float(line.split(":")[1].strip())

            if avg_time is not None:
                speedup = seq_baseline[size] / avg_time
                efficiency = (speedup / p) * 100

                print(f"Selesai! (Rata-rata: {avg_time:.6f}s, Speedup: {speedup:.2f}x)")

                results_raw.append({
                    "method": "MPI",
                    "matrix_size": size,
                    "processes": p,
                    "repeats": repeats,
                    "average_time": f"{avg_time:.9f}",
                    "min_time": f"{min_time:.9f}",
                    "checksum": f"{checksum:.9f}"
                })

                metrics.append({
                    "matrix_size": size,
                    "processes": p,
                    "execution_time": f"{avg_time:.9f}",
                    "speedup": speedup,
                    "efficiency_percent": efficiency
                })
            else:
                print(f"Gagal mem-parsing output! Output:\n{output}")

        except Exception as e:
            print(f"Error saat eksekusi: {e}")

# Simpan ke CSV
results_path = os.path.join(RESULTS_DIR, "results_mpi.csv")
with open(results_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "matrix_size", "processes", "repeats", "average_time", "min_time", "checksum"])
    writer.writeheader()
    writer.writerows(results_raw)

metrics_path = os.path.join(RESULTS_DIR, "mpi_metrics.csv")
with open(metrics_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["matrix_size", "processes", "execution_time", "speedup", "efficiency_percent"])
    writer.writeheader()
    writer.writerows(metrics)

print("=" * 65)
print(f"Hasil mentah berhasil disimpan di: {results_path}")
print(f"Hasil metrik berhasil disimpan di: {metrics_path}")
print("=" * 65)
