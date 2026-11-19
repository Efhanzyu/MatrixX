import csv
import os
import matplotlib.pyplot as plt

# Folder proyek
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
ANGGOTA1_RESULTS = os.path.join(ROOT_DIR, "anggota1", "results", "results_sequential.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)


def read_csv(path):
    with open(path, "r", newline="") as file:
        return list(csv.DictReader(file))


# Baca data MPI
mpi_results_file = os.path.join(RESULTS_DIR, "results_mpi.csv")
if not os.path.exists(mpi_results_file):
    print(f"Error: {mpi_results_file} belum ada. Jalankan benchmark terlebih dahulu!")
    exit(1)

mpi_data_raw = read_csv(mpi_results_file)

# Baca data Sequential Anggota 1
if os.path.exists(ANGGOTA1_RESULTS):
    seq_data_raw = read_csv(ANGGOTA1_RESULTS)
    seq_data = {
        int(row["matrix_size"]): float(row["average_time"])
        for row in seq_data_raw
    }
else:
    seq_data = {
        512: 0.033638255,
        1024: 0.452294608,
        2048: 4.996364580,
        4096: 37.596399516
    }

mpi_data = {}
for row in mpi_data_raw:
    size = int(row["matrix_size"])
    procs = int(row["processes"])
    mpi_data[(size, procs)] = float(row["average_time"])

matrix_sizes = [512, 1024, 2048, 4096]
process_counts = [1, 2, 4, 8]

# 1. Grafik: Execution Time Comparison
plt.figure(figsize=(10, 6))
for size in matrix_sizes:
    times = [mpi_data[(size, p)] for p in process_counts]
    plt.plot(process_counts, times, marker="o", linewidth=2, label=f"{size} x {size}")

plt.xlabel("Number of MPI Processes", fontsize=12)
plt.ylabel("Average Execution Time (seconds)", fontsize=12)
plt.yscale("log")
plt.title("MPI Matrix Multiplication Execution Time", fontsize=14, fontweight="bold")
plt.xticks(process_counts)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Matrix Size")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "mpi_execution_time.png"), dpi=300)
plt.close()

# 2. Grafik: Speedup relative to Sequential
plt.figure(figsize=(10, 6))
for size in matrix_sizes:
    speedups = [seq_data[size] / mpi_data[(size, p)] for p in process_counts]
    plt.plot(process_counts, speedups, marker="s", linewidth=2, label=f"{size} x {size}")

plt.plot(process_counts, process_counts, linestyle="--", color="gray", label="Ideal Linear Speedup")
plt.xlabel("Number of MPI Processes", fontsize=12)
plt.ylabel("Speedup", fontsize=12)
plt.title("MPI Speedup Relative to Sequential Baseline", fontsize=14, fontweight="bold")
plt.xticks(process_counts)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Matrix Size")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "mpi_speedup.png"), dpi=300)
plt.close()

# 3. Grafik: Parallel Efficiency
plt.figure(figsize=(10, 6))
for size in matrix_sizes:
    efficiencies = [
        ((seq_data[size] / mpi_data[(size, p)]) / p) * 100
        for p in process_counts
    ]
    plt.plot(process_counts, efficiencies, marker="^", linewidth=2, label=f"{size} x {size}")

plt.axhline(100, linestyle="--", color="gray", label="Ideal Efficiency (100%)")
plt.xlabel("Number of MPI Processes", fontsize=12)
plt.ylabel("Parallel Efficiency (%)", fontsize=12)
plt.title("MPI Parallel Efficiency", fontsize=14, fontweight="bold")
plt.xticks(process_counts)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Matrix Size")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "mpi_efficiency.png"), dpi=300)
plt.close()

print("=================================================")
print("          ANALISIS PERFORMA MPI MATRIXX          ")
print("=================================================")
for size in matrix_sizes:
    print(f"\n--- Matriks {size} x {size} (Baseline Seq: {seq_data[size]:.6f}s) ---")
    for p in process_counts:
        t = mpi_data[(size, p)]
        sp = seq_data[size] / t
        eff = (sp / p) * 100
        print(f"  MPI {p} Proses: Waktu = {t:10.6f}s | Speedup = {sp:5.2f}x | Efisiensi = {eff:6.2f}%")

print("\nSeluruh grafik berhasil disimpan di folder 'anggota2/figures/'!")
