#!/usr/bin/env python3
"""Benchmark Hybrid MPI-OpenMP (Anggota 3) + baseline Sequential lokal."""
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
HYB_BIN = os.path.join(BASE_DIR, "hybrid")

# (proses MPI, thread per proses); total thread <= 4 core laptop
CONFIGS = [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1)]
TOLERANCE = 1e-12  # toleransi relatif checksum
LARGE_SIZE = 2048  # ukuran di atas ini: baseline hanya 1 run (terlalu lama)


def parse_output(text):
    values = {}
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            values[key.strip()] = val.strip().split()[0] if val.strip() else ""
    return values


def run_sequential(size, repeats):
    res = subprocess.run([SEQ_BIN, str(size), str(repeats)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, check=True)
    return parse_output(res.stdout)


def run_hybrid(size, repeats, procs, threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    cmd = ["mpirun", "--bind-to", "none", "-np", str(procs),
           HYB_BIN, str(size), str(repeats)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, check=True, env=env)
    return parse_output(res.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--baseline-runs", type=int, default=3,
                    help="jumlah run terpisah baseline untuk ukuran <= 2048")
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "results"))
    args = ap.parse_args()

    for path in (SEQ_BIN, HYB_BIN):
        if not os.path.exists(path):
            sys.exit(f"Binary tidak ditemukan: {path} (kompilasi dulu)")

    os.makedirs(args.out_dir, exist_ok=True)

    print("Pemanasan CPU (hasil dibuang)...", flush=True)
    run_sequential(1024, 2)
    run_hybrid(512, 2, 1, 1)

    seq_path = os.path.join(args.out_dir, "results_sequential_local.csv")
    hyb_path = os.path.join(args.out_dir, "results_hybrid.csv")
    met_path = os.path.join(args.out_dir, "hybrid_metrics.csv")

    seq_avg, seq_min, seq_sum = {}, {}, {}

    with open(seq_path, "w", newline="") as fs:
        ws = csv.writer(fs)
        ws.writerow(["method", "matrix_size", "threads", "repeats",
                     "baseline_runs", "average_time", "min_time", "checksum"])
        for size in args.sizes:
            runs = args.baseline_runs if size <= LARGE_SIZE else 1
            avgs, mins = [], []
            chk = None
            for r in range(runs):
                print(f"[Sequential lokal] {size}x{size} run {r + 1}/{runs} ...",
                      end=" ", flush=True)
                v = run_sequential(size, args.repeats)
                avgs.append(float(v["Average Execution Time"]))
                mins.append(float(v["Minimum Execution Time"]))
                chk = v["Checksum"]
                print(f"{avgs[-1]:.6f}s", flush=True)
            seq_avg[size] = statistics.median(avgs)
            seq_min[size] = min(mins)
            seq_sum[size] = chk
            ws.writerow(["Sequential", size, 1, args.repeats, runs,
                         f"{seq_avg[size]:.9f}", f"{seq_min[size]:.9f}", chk])
            fs.flush()
            print(f"  -> baseline {size}: median rata-rata {seq_avg[size]:.6f}s "
                  f"(semua run: {', '.join(f'{a:.4f}' for a in avgs)})", flush=True)

    total = len(args.sizes) * len(CONFIGS)
    idx = 0

    with open(hyb_path, "w", newline="") as fh, \
         open(met_path, "w", newline="") as fm:
        wh = csv.writer(fh)
        wm = csv.writer(fm)
        wh.writerow(["method", "matrix_size", "processes", "threads_per_process",
                     "total_threads", "repeats", "average_time", "min_time",
                     "compute_time", "checksum", "checksum_match"])
        wm.writerow(["matrix_size", "processes", "threads_per_process",
                     "total_threads", "execution_time", "compute_time",
                     "comm_overhead", "speedup", "efficiency_percent",
                     "speedup_min"])

        for size in args.sizes:
            for procs, threads in CONFIGS:
                idx += 1
                print(f"[{idx}/{total}] Hybrid {size}x{size}, "
                      f"MPI {procs} x OpenMP {threads} ...", end=" ", flush=True)
                t0 = time.time()
                v = run_hybrid(size, args.repeats, procs, threads)

                avg = float(v["Average Execution Time"])
                mn = float(v["Minimum Execution Time"])
                comp = float(v["Average Compute Time"])
                total_threads = int(v["Total Threads"])
                chk = v["Checksum"]

                ref = float(seq_sum[size])
                match = abs(float(chk) - ref) <= TOLERANCE * abs(ref)
                if total_threads != procs * threads:
                    print(f"\n  PERINGATAN: total thread aktual {total_threads} "
                          f"!= {procs * threads}", flush=True)

                speedup = seq_avg[size] / avg
                efficiency = speedup / total_threads * 100.0
                speedup_min = seq_min[size] / mn

                wh.writerow(["Hybrid", size, procs, threads, total_threads,
                             args.repeats, f"{avg:.9f}", f"{mn:.9f}",
                             f"{comp:.9f}", chk, "yes" if match else "NO"])
                wm.writerow([size, procs, threads, total_threads,
                             f"{avg:.9f}", f"{comp:.9f}", f"{avg - comp:.9f}",
                             f"{speedup:.6f}", f"{efficiency:.4f}",
                             f"{speedup_min:.6f}"])
                fh.flush()
                fm.flush()
                print(f"{avg:.6f}s, speedup {speedup:.2f}x, "
                      f"checksum {'OK' if match else 'BEDA!'} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    print("Selesai. Hasil di:", args.out_dir)


if __name__ == "__main__":
    main()
