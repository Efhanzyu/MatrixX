
import csv
import os
import matplotlib.pyplot as plt

# Lokasi folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Membaca data validasi
csv_path = os.path.join(RESULTS_DIR, "results_validation.csv")

with open(csv_path, "r", newline="") as file:
    data = list(csv.DictReader(file))

# Memisahkan data Sequential dan OpenMP
sequential = {}
openmp = {}

for row in data:
    size = int(row["matrix_size"])
    method = row["method"]

    if method == "Sequential":
        sequential[size] = float(row["average_time"])

    elif method == "OpenMP":
        threads = int(row["threads"])
        openmp[(size, threads)] = float(row["average_time"])

matrix_sizes = [1024, 4096]
thread_counts = [1, 2, 4, 8]

# Menghitung speedup dan efisiensi
metrics = []

for size in matrix_sizes:
    baseline = sequential[size]

    for threads in thread_counts:
        execution_time = openmp[(size, threads)]
        speedup = baseline / execution_time
        efficiency = speedup / threads * 100

        metrics.append({
            "matrix_size": size,
            "threads": threads,
            "sequential_time": baseline,
            "openmp_time": execution_time,
            "speedup": speedup,
            "efficiency_percent": efficiency
        })

# Menyimpan tabel metrik validasi
metrics_path = os.path.join(
    RESULTS_DIR, "validation_metrics.csv"
)

with open(metrics_path, "w", newline="") as file:
    fields = [
        "matrix_size",
        "threads",
        "sequential_time",
        "openmp_time",
        "speedup",
        "efficiency_percent"
    ]

    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(metrics)


# Grafik 1: Validation Execution Time
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, size in zip(axes, matrix_sizes):
    times = [openmp[(size, t)] for t in thread_counts]
    baseline = sequential[size]

    ax.plot(
        thread_counts,
        times,
        marker="o",
        linewidth=2,
        label="OpenMP"
    )

    ax.axhline(
        baseline,
        linestyle="--",
        label="Sequential"
    )

    ax.set_title(f"Matrix {size} x {size}")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Average Execution Time (seconds)")
    ax.set_xticks(thread_counts)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

fig.suptitle("Validation Execution Time Comparison")
fig.tight_layout()

fig.savefig(
    os.path.join(FIGURES_DIR, "validation_execution_time.png"),
    dpi=300
)
plt.close(fig)


# Grafik 2: Validation Speedup
plt.figure(figsize=(9, 6))

for size in matrix_sizes:
    values = [
        sequential[size] / openmp[(size, t)]
        for t in thread_counts
    ]

    plt.plot(
        thread_counts,
        values,
        marker="o",
        linewidth=2,
        label=f"{size} x {size}"
    )

plt.plot(
    thread_counts,
    thread_counts,
    linestyle="--",
    label="Ideal Scaling"
)

plt.xlabel("Number of Threads")
plt.ylabel("Speedup Relative to Sequential")
plt.title("Validation OpenMP Speedup")
plt.xticks(thread_counts)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Matrix Size")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "validation_speedup.png"),
    dpi=300
)
plt.close()


# Grafik 3: Validation Parallel Efficiency
plt.figure(figsize=(9, 6))

for size in matrix_sizes:
    values = [
        (sequential[size] / openmp[(size, t)]) / t * 100
        for t in thread_counts
    ]

    plt.plot(
        thread_counts,
        values,
        marker="o",
        linewidth=2,
        label=f"{size} x {size}"
    )

plt.axhline(
    100,
    linestyle="--",
    label="Ideal Efficiency"
)

plt.xlabel("Number of Threads")
plt.ylabel("Parallel Efficiency (%)")
plt.title("Validation OpenMP Parallel Efficiency")
plt.xticks(thread_counts)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Matrix Size")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "validation_efficiency.png"),
    dpi=300
)
plt.close()


# Menampilkan hasil analisis
print("==========================================")
print("VALIDATION PERFORMANCE ANALYSIS")
print("==========================================")

for size in matrix_sizes:
    print(f"\nMatrix: {size} x {size}")
    print(f"Sequential: {sequential[size]:.9f} seconds")

    for threads in thread_counts:
        time = openmp[(size, threads)]
        speedup = sequential[size] / time
        efficiency = speedup / threads * 100

        print(
            f"OpenMP {threads} thread(s): "
            f"{time:.9f} s | "
            f"Speedup: {speedup:.2f}x | "
            f"Efficiency: {efficiency:.2f}%"
        )

print("\n==========================================")
print("Validation analysis completed.")
print("Metrics: results/validation_metrics.csv")
print("Figures: figures/")
print("==========================================")
