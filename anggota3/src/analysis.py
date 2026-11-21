#!/usr/bin/env python3
"""Grafik dan ringkasan hasil Hybrid MPI-OpenMP (Anggota 3).

Membaca results/hybrid_metrics.csv dan menulis grafik ke figures/.
Semua angka berasal dari CSV hasil benchmark; tidak ada nilai yang ditulis manual.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
METRICS_FILE = os.path.join(RESULTS_DIR, "hybrid_metrics.csv")
SEQ_FILE = os.path.join(RESULTS_DIR, "results_sequential_local.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)


def read_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


for path in (METRICS_FILE, SEQ_FILE):
    if not os.path.exists(path):
        raise SystemExit(f"Error: {path} belum ada. Jalankan benchmark terlebih dahulu!")

rows = read_csv(METRICS_FILE)
seq_rows = read_csv(SEQ_FILE)

seq = {int(r["matrix_size"]): float(r["average_time_median"]) for r in seq_rows}

data = {}
for r in rows:
    key = (int(r["matrix_size"]), int(r["processes"]), int(r["threads_per_process"]))
    data[key] = {
        "total_threads": int(r["total_threads"]),
        "time": float(r["execution_time"]),
        "compute": float(r["compute_time"]),
        "comm": float(r["comm_overhead"]),
        "speedup": float(r["speedup"]),
        "eff": float(r["efficiency_percent"]),
        "speedup_rel": float(r["speedup_vs_hybrid1x1"]),
    }

sizes = sorted({k[0] for k in data})
configs = sorted({(k[1], k[2]) for k in data}, key=lambda c: (c[0] * c[1], c[0]))
labels = [f"MPI {p} x OMP {t}" for p, t in configs]
cmap = plt.get_cmap("viridis")
colors = [cmap(i / max(len(configs) - 1, 1)) for i in range(len(configs))]


def grouped_bars(metric, ylabel, title, filename, log=False, hline=None, extra_line=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    n = len(configs)
    width = 0.8 / n
    for ci, (p, t) in enumerate(configs):
        vals = [data[(s, p, t)][metric] for s in sizes]
        xs = [i + (ci - (n - 1) / 2) * width for i in range(len(sizes))]
        ax.bar(xs, vals, width=width, color=colors[ci], label=labels[ci])
    if hline is not None:
        ax.axhline(hline[0], linestyle="--", color="gray", label=hline[1])
    if extra_line is not None:
        xs = list(range(len(sizes)))
        ax.plot(xs, [extra_line[0][s] for s in sizes], marker="o", color="red",
                linewidth=2, label=extra_line[1])
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s} x {s}" for s in sizes])
    ax.set_xlabel("Matrix Size", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(title="Configuration", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    plt.close(fig)


# 1. Waktu eksekusi (dengan baseline Sequential lokal sebagai garis pembanding)
grouped_bars("time", "Average Execution Time (seconds)",
             "Hybrid MPI-OpenMP Execution Time",
             "hybrid_execution_time.png", log=True,
             extra_line=(seq, "Sequential (local baseline)"))

# 2. Speedup terhadap Sequential lokal
grouped_bars("speedup", "Speedup (T_sequential / T_hybrid)",
             "Hybrid MPI-OpenMP Speedup Relative to Local Sequential Baseline",
             "hybrid_speedup.png", hline=(1.0, "Speedup = 1 (equal to Sequential)"))

# 3. Efisiensi paralel: Speedup / total thread x 100%
grouped_bars("eff", "Parallel Efficiency (%)",
             "Hybrid MPI-OpenMP Parallel Efficiency (P = processes x threads)",
             "hybrid_efficiency.png", hline=(100.0, "Ideal Efficiency (100%)"))

# 4. Tambahan: pemisahan waktu komputasi dan overhead komunikasi
fig, ax = plt.subplots(figsize=(12, 6))
n = len(configs)
width = 0.8 / n
for ci, (p, t) in enumerate(configs):
    comp = [data[(s, p, t)]["compute"] for s in sizes]
    comm = [data[(s, p, t)]["comm"] for s in sizes]
    xs = [i + (ci - (n - 1) / 2) * width for i in range(len(sizes))]
    ax.bar(xs, comp, width=width, color=colors[ci], label=labels[ci])
    ax.bar(xs, comm, width=width, bottom=comp, color="red", alpha=0.7,
           label="Communication / other overhead" if ci == 0 else None)
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels([f"{s} x {s}" for s in sizes])
ax.set_xlabel("Matrix Size", fontsize=12)
ax.set_ylabel("Average Time (seconds)", fontsize=12)
ax.set_yscale("log")
ax.set_title("Hybrid Time Breakdown: Compute vs. Communication Overhead",
             fontsize=14, fontweight="bold")
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Configuration", ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "hybrid_time_breakdown.png"), dpi=300)
plt.close(fig)

# Ringkasan teks (semua dari CSV)
print("=" * 78)
print("             RINGKASAN PERFORMA HYBRID MPI-OpenMP (ANGGOTA 3)")
print("=" * 78)
for s in sizes:
    print(f"\n--- Matriks {s} x {s} (Baseline Sequential lokal: {seq[s]:.6f}s) ---")
    best = max(configs, key=lambda c: data[(s, c[0], c[1])]["speedup"])
    for p, t in configs:
        d = data[(s, p, t)]
        mark = "  <- speedup tertinggi" if (p, t) == best else ""
        print(f"  MPI {p} x OMP {t} (P={d['total_threads']}): "
              f"waktu={d['time']:10.6f}s | speedup={d['speedup']:5.2f}x | "
              f"efisiensi={d['eff']:6.2f}% | overhead={d['comm']:.6f}s{mark}")
print(f"\nGrafik tersimpan di: {FIGURES_DIR}")
