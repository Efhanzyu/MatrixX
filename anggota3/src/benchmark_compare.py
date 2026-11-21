#!/usr/bin/env python3
"""Pembanding lokal (Anggota 3): OpenMP (Anggota 1) dan MPI (Anggota 2)
dijalankan di laptop yang sama dengan Hybrid, memakai source code mereka
tanpa perubahan. Baseline Sequential diambil dari results_sequential_local.csv
(hasil benchmark.py) agar semua metode dibandingkan dengan baseline yang sama.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
OMP_BIN = os.path.join(ROOT_DIR, "anggota1", "openmp")
MPI_BIN = os.path.join(ROOT_DIR, "anggota2", "mpi")
SEQ_CSV = os.path.join(BASE_DIR, "results", "results_sequential_local.csv")

UNITS = [1, 2, 4]  # thread (OpenMP) atau proses (MPI); <= 4 core laptop
TOLERANCE = 1e-12


def parse_output(text):
    values = {}
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            values[key.strip()] = val.strip().split()[0] if val.strip() else ""
    return values


def run_openmp(size, repeats, threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    res = subprocess.run([OMP_BIN, str(size), str(repeats)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, check=True, env=env)
    return parse_output(res.stdout)


def run_mpi(size, repeats, procs):
    cmd = ["mpirun", "--bind-to", "none", "-np", str(procs),
           MPI_BIN, str(size), str(repeats)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, check=True)
    return parse_output(res.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=["OpenMP", "MPI"],
                    choices=["OpenMP", "MPI"])
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "results"))
    args = ap.parse_args()

    if not os.path.exists(SEQ_CSV):
        sys.exit(f"Baseline tidak ditemukan: {SEQ_CSV} (jalankan benchmark.py dulu)")
    needed = {"OpenMP": OMP_BIN, "MPI": MPI_BIN}
    for m in args.methods:
        if not os.path.exists(needed[m]):
            sys.exit(f"Binary tidak ditemukan: {needed[m]} (kompilasi dulu)")

    seq_avg, seq_chk = {}, {}
    with open(SEQ_CSV, newline="") as f:
        for r in csv.DictReader(f):
            seq_avg[int(r["matrix_size"])] = float(r["average_time_median"])
            seq_chk[int(r["matrix_size"])] = float(r["checksum"])
    for s in args.sizes:
        if s not in seq_avg:
            sys.exit(f"Baseline untuk ukuran {s} tidak ada di {SEQ_CSV}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "comparison_local.csv")

    print("Pemanasan CPU (hasil dibuang)...", flush=True)
    if "OpenMP" in args.methods:
        run_openmp(512, 2, 1)
    if "MPI" in args.methods:
        run_mpi(512, 2, 1)

    total = len(args.sizes) * len(args.methods) * len(UNITS)
    idx = 0
    with open(out_path, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["method", "matrix_size", "units", "repeats", "average_time",
                    "min_time", "checksum", "checksum_match", "speedup",
                    "efficiency_percent"])
        for size in args.sizes:
            for method in args.methods:
                for u in UNITS:
                    idx += 1
                    print(f"[{idx}/{total}] {method} {size}x{size}, P={u} ...",
                          end=" ", flush=True)
                    t0 = time.time()
                    v = (run_openmp(size, args.repeats, u) if method == "OpenMP"
                         else run_mpi(size, args.repeats, u))
                    actual = int(v["Threads"] if method == "OpenMP" else v["Processes"])
                    if actual != u:
                        print(f"\n  PERINGATAN: unit aktual {actual} != {u}", flush=True)
                    avg = float(v["Average Execution Time"])
                    mn = float(v["Minimum Execution Time"])
                    chk = v["Checksum"]
                    ref = seq_chk[size]
                    match = abs(float(chk) - ref) <= TOLERANCE * abs(ref)
                    sp = seq_avg[size] / avg
                    w.writerow([method, size, u, args.repeats, f"{avg:.9f}",
                                f"{mn:.9f}", chk, "yes" if match else "NO",
                                f"{sp:.6f}", f"{sp / u * 100:.4f}"])
                    fo.flush()
                    print(f"{avg:.6f}s, speedup {sp:.2f}x, "
                          f"checksum {'OK' if match else 'BEDA!'} "
                          f"({time.time() - t0:.0f}s)", flush=True)
    print("Selesai. Hasil di:", out_path)


if __name__ == "__main__":
    main()
