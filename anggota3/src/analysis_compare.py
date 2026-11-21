#!/usr/bin/env python3
"""Grafik perbandingan antarmetode dari uji bergantian (Anggota 3).

Sumber data: results/interleaved_summary.csv (1024, 2048) dan
results/interleaved_4096/interleaved_summary.csv (4096). Batang = median
antarputaran; garis error = rentang minimum-maksimum antarputaran.
Semua angka dibaca dari CSV; tidak ada nilai manual.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE_DIR, "results")
FIG = os.path.join(BASE_DIR, "figures")
SOURCES = [os.path.join(RES, "interleaved_summary.csv"),
           os.path.join(RES, "interleaved_4096", "interleaved_summary.csv")]
METHODS = ["Sequential", "OpenMP 4 thread", "MPI 4 proses",
           "Hybrid 1x4", "Hybrid 2x2"]
UNITS = {"Sequential": 1, "OpenMP 4 thread": 4, "MPI 4 proses": 4,
         "Hybrid 1x4": 4, "Hybrid 2x2": 4}

os.makedirs(FIG, exist_ok=True)

data, rounds = {}, {}
for path in SOURCES:
    if not os.path.exists(path):
        print(f"Peringatan: {path} tidak ada, dilewati.")
        continue
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["matrix_size"]), r["method"])
            data[key] = (float(r["median_average_time"]),
                         float(r["min_of_averages"]),
                         float(r["max_of_averages"]))
            rounds[int(r["matrix_size"])] = int(r["rounds"])

sizes = sorted({k[0] for k in data})
if not sizes:
    raise SystemExit("Error: tidak ada data uji bergantian. Jalankan benchmark_interleaved.py dulu.")
colors = plt.get_cmap("viridis")([i / (len(METHODS) - 1) for i in range(len(METHODS))])
note = ", ".join(f"{s}: {rounds[s]} putaran" for s in sizes)


def bars(ax, values, errors=None):
    n = len(METHODS)
    w = 0.8 / n
    for mi, m in enumerate(METHODS):
        xs = [i + (mi - (n - 1) / 2) * w for i in range(len(sizes))]
        vals = [values(s, m) for s in sizes]
        err = None
        if errors:
            err = [[max(vals[i] - errors(s, m)[0], 0) for i, s in enumerate(sizes)],
                   [max(errors(s, m)[1] - vals[i], 0) for i, s in enumerate(sizes)]]
        ax.bar(xs, vals, width=w, color=colors[mi], label=m, yerr=err, capsize=3)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s} x {s}" for s in sizes])
    ax.set_xlabel("Matrix Size")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)


# 1. Waktu eksekusi
fig, ax = plt.subplots(figsize=(11, 6))
bars(ax, lambda s, m: data[(s, m)][0],
     lambda s, m: (data[(s, m)][1], data[(s, m)][2]))
ax.set_ylabel("Average Execution Time (seconds)")
ax.set_yscale("log")
ax.set_title("Method Comparison (interleaved runs, 4 processing units)\n"
             f"median, bars = min-max across rounds ({note})", fontsize=12)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "compare_execution_time.png"), dpi=300)
plt.close(fig)

# 2. Speedup dan efisiensi (terhadap Sequential dari uji bergantian yang sama)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
sp = lambda s, m: data[(s, "Sequential")][0] / data[(s, m)][0]  # noqa: E731
bars(a1, sp)
a1.axhline(1.0, linestyle="--", color="gray")
a1.axhline(4.0, linestyle=":", color="red", label="Ideal (4 units)")
a1.set_ylabel("Speedup (T_sequential / T_method)")
a1.set_title("Speedup", fontweight="bold")
a1.legend(fontsize=8)
bars(a2, lambda s, m: sp(s, m) / UNITS[m] * 100)
a2.axhline(100.0, linestyle="--", color="gray")
a2.set_ylabel("Parallel Efficiency (%)")
a2.set_title("Parallel Efficiency (speedup / P x 100%)", fontweight="bold")
fig.suptitle("Method Comparison vs. Same-Session Sequential Baseline", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "compare_speedup_efficiency.png"), dpi=300)
plt.close(fig)

print("Ringkasan (median antarputaran):")
for s in sizes:
    print(f"\n--- {s}x{s} ({rounds[s]} putaran) ---")
    for m in METHODS:
        med, lo, hi = data[(s, m)]
        print(f"  {m:<16} median={med:10.6f}s rentang=[{lo:.4f}, {hi:.4f}] "
              f"speedup={sp(s, m):5.2f}x efisiensi={sp(s, m) / UNITS[m] * 100:6.2f}%")
print(f"\nGrafik tersimpan di: {FIG}")
