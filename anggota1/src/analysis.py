
import csv
import os
import matplotlib.pyplot as plt

# Folder proyek
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


def read_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r", newline="") as file:
        return list(csv.DictReader(file))


# Membaca data benchmark awal
sequential = read_csv("results_sequential.csv")
openmp = read_csv("results_openmp.csv")

seq_data = {
    int(row["matrix_size"]): float(row["average_time"])
    for row in sequential
}

omp_data = {}
for row in openmp:
    size = int(row["matrix_size"])
    threads = int(row["threads"])
    omp_data[(size, threads)] = float(row["average_time"])


matrix_sizes = [512, 1024, 2048, 4096]
thread_counts = [1, 2, 4, 8]


# Menghitung speedup dan efisiensi
metrics = []

for size in matrix_sizes:
    baseline = seq_data[size]

    for threads in thread_counts:
        execution_time = omp_data[(size, threads)]
        speedup = baseline / execution_time
        efficiency = speedup / threads * 100

        metrics.append({
            "matrix_size": size,
            "threads": threads,
            "execution_time": execution_time,
            "speedup": speedup,
            "efficiency_percent": efficiency
        })


# Menyimpan hasil perhitungan metrik
metrics_path = os.path.join(RESULTS_DIR, "performance_metrics.csv")

with open(metrics_path, "w", newline="") as file:
    fields = [
        "matrix_size",
        "threads",
        "execution_time",
        "speedup",
        "efficiency_percent"
    ]

    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(metrics)


# Grafik 1: Execution Time Comparison
plt.figure(figsize=(10, 6))

for size in matrix_sizes:
    times = [omp_data[(size, t)] for t in thread_counts]
    plt.plot(thread_counts, times, marker="o", label=f"{size} x {size}")

plt.xlabel("Number of Threads")
plt.ylabel("Average Execution Time (seconds)")
plt.yscale("log")
plt.title("OpenMP Execution Time Comparison")
plt.xticks(thread_counts)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Matrix Size")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "execution_time.png"), dpi=300)
plt.close()


# Grafik 2: Strong Scaling
plt.figure(figsize=(10, 6))

for size in matrix_sizes:
    times = [omp_data[(size, t)] for t in thread_counts]
    speedup = [times[0] / value for value in times]

    plt.plot(thread_counts, speedup, marker="o", label=f"{size} x {size}")

plt.plot(thread_counts, thread_counts, linestyle="--",
         label="Ideal Scaling")

plt.xlabel("Number of Threads")
plt.ylabel("Speedup Relative to OpenMP 1 Thread")
plt.title("OpenMP Strong Scaling")
plt.xticks(thread_counts)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "strong_scaling.png"), dpi=300)
plt.close()


# Grafik 3: Speedup terhadap Sequential
plt.figure(figsize=(10, 6))

for size in matrix_sizes:
    values = [
        seq_data[size] / omp_data[(size, t)]
        for t in thread_counts
    ]

    plt.plot(thread_counts, values, marker="o", label=f"{size} x {size}")

plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("OpenMP Speedup Relative to Sequential")
plt.xticks(thread_counts)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Matrix Size")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "speedup.png"), dpi=300)
plt.close()


# Grafik 4: Parallel Efficiency
plt.figure(figsize=(10, 6))

for size in matrix_sizes:
    values = [
        (seq_data[size] / omp_data[(size, t)]) / t * 100
        for t in thread_counts
    ]

    plt.plot(thread_counts, values, marker="o", label=f"{size} x {size}")

plt.axhline(100, linestyle="--", label="Ideal Efficiency")
plt.xlabel("Number of Threads")
plt.ylabel("Parallel Efficiency (%)")
plt.title("OpenMP Parallel Efficiency")
plt.xticks(thread_counts)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Matrix Size")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "parallel_efficiency.png"), dpi=300)
plt.close()


print("======================================")
print("ANALISIS PERFORMA MATRIXX")
print("======================================")
print(f"Jumlah ukuran matriks : {len(matrix_sizes)}")
print(f"Jumlah konfigurasi    : {len(metrics)}")
print("File metrik           : results/performance_metrics.csv")
print("Folder grafik         : figures/")
print("--------------------------------------")

for size in matrix_sizes:
    print(f"\nMatriks {size} x {size}")
    print(f"Sequential: {seq_data[size]:.6f} detik")

    for threads in thread_counts:
        time = omp_data[(size, threads)]
        speedup = seq_data[size] / time
        efficiency = speedup / threads * 100

        print(
            f"OpenMP {threads:2d} thread: "
            f"{time:.6f} s | "
            f"Speedup: {speedup:.2f}x | "
            f"Efisiensi: {efficiency:.2f}%"
        )

print("\nSemua grafik berhasil dibuat.")
