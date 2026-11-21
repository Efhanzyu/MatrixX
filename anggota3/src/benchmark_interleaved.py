#!/usr/bin/env python3
"""Uji bergantian lintas metode (Anggota 3).

Sequential, OpenMP, MPI, dan Hybrid dijalankan selang-seling dalam satu sesi,
beberapa putaran, dengan urutan diputar tiap putaran. Tujuannya mengurangi
pengaruh drift kondisi laptop pada perbandingan antarmetode. Semua metode
memakai total 4 unit pemrosesan (= 4 core laptop), kecuali Sequential.
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
SEQ_BIN = os.path.join(ROOT_DIR, "anggota1", "sequential")
OMP_BIN = os.path.join(ROOT_DIR, "anggota1", "openmp")
MPI_BIN = os.path.join(ROOT_DIR, "anggota2", "mpi")
HYB_BIN = os.path.join(BASE_DIR, "hybrid")
SEQ_CSV = os.path.join(BASE_DIR, "results", "results_sequential_local.csv")
TOLERANCE = 1e-12


def parse_output(text):
    values = {}
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            values[key.strip()] = val.strip().split()[0] if val.strip() else ""
    return values


def mpi_cmd(binary, procs, size, repeats):
    return ["mpirun", "--bind-to", "none", "-np", str(procs),
            binary, str(size), str(repeats)]


# (label, fungsi(size, repeats) -> (cmd, extra_env))
METHODS = [
    ("Sequential", lambda s, r: ([SEQ_BIN, str(s), str(r)], {})),
    ("OpenMP 4 thread", lambda s, r: ([OMP_BIN, str(s), str(r)],
                                      {"OMP_NUM_THREADS": "4"})),
    ("MPI 4 proses", lambda s, r: (mpi_cmd(MPI_BIN, 4, s, r), {})),
    ("Hybrid 1x4", lambda s, r: (mpi_cmd(HYB_BIN, 1, s, r),
                                 {"OMP_NUM_THREADS": "4"})),
    ("Hybrid 2x2", lambda s, r: (mpi_cmd(HYB_BIN, 2, s, r),
                                 {"OMP_NUM_THREADS": "2"})),
]


def run(cmd, extra_env):
    env = os.environ.copy()
    env.update(extra_env)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, check=True, env=env)
    return parse_output(res.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "results"))
    args = ap.parse_args()

    for path in (SEQ_BIN, OMP_BIN, MPI_BIN, HYB_BIN, SEQ_CSV):
        if not os.path.exists(path):
            sys.exit(f"File tidak ditemukan: {path}")

    ref_chk = {}
    with open(SEQ_CSV, newline="") as f:
        for r in csv.DictReader(f):
            ref_chk[int(r["matrix_size"])] = float(r["checksum"])
    for s in args.sizes:
        if s not in ref_chk:
            sys.exit(f"Checksum referensi ukuran {s} tidak ada di {SEQ_CSV}")

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "interleaved_comparison.csv")
    sum_path = os.path.join(args.out_dir, "interleaved_summary.csv")

    print("Pemanasan CPU (hasil dibuang)...", flush=True)
    run(*METHODS[0][1](512, 2))
    run(*METHODS[3][1](512, 2))

    results = {}  # (size, label) -> list rata-rata per putaran
    n = len(METHODS)
    total = len(args.sizes) * args.rounds * n
    idx = 0
    with open(raw_path, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["matrix_size", "round", "order_in_round", "method",
                    "repeats", "average_time", "min_time", "checksum",
                    "checksum_match"])
        for size in args.sizes:
            for rnd in range(args.rounds):
                order = METHODS[rnd % n:] + METHODS[:rnd % n]
                for pos, (label, builder) in enumerate(order, start=1):
                    idx += 1
                    print(f"[{idx}/{total}] {size}x{size} putaran {rnd + 1} "
                          f"#{pos} {label} ...", end=" ", flush=True)
                    t0 = time.time()
                    v = run(*builder(size, args.repeats))
                    avg = float(v["Average Execution Time"])
                    mn = float(v["Minimum Execution Time"])
                    chk = v["Checksum"]
                    ref = ref_chk[size]
                    match = abs(float(chk) - ref) <= TOLERANCE * abs(ref)
                    w.writerow([size, rnd + 1, pos, label, args.repeats,
                                f"{avg:.9f}", f"{mn:.9f}", chk,
                                "yes" if match else "NO"])
                    fo.flush()
                    results.setdefault((size, label), []).append(avg)
                    print(f"{avg:.6f}s, checksum "
                          f"{'OK' if match else 'BEDA!'} "
                          f"({time.time() - t0:.0f}s)", flush=True)

    with open(sum_path, "w", newline="") as fs:
        w = csv.writer(fs)
        w.writerow(["matrix_size", "method", "rounds", "median_average_time",
                    "min_of_averages", "max_of_averages",
                    "speedup_vs_sequential_median"])
        print("\n=== RINGKASAN (median antarputaran) ===")
        for size in args.sizes:
            seq_med = statistics.median(results[(size, "Sequential")])
            print(f"\n--- {size}x{size} ---")
            for label, _ in METHODS:
                vals = results[(size, label)]
                med = statistics.median(vals)
                sp = seq_med / med
                w.writerow([size, label, len(vals), f"{med:.9f}",
                            f"{min(vals):.9f}", f"{max(vals):.9f}",
                            f"{sp:.6f}"])
                print(f"  {label:<16} median={med:10.6f}s  "
                      f"rentang=[{min(vals):.4f}, {max(vals):.4f}]  "
                      f"speedup={sp:5.2f}x")
    print("\nSelesai. Hasil di:", args.out_dir)


if __name__ == "__main__":
    main()
