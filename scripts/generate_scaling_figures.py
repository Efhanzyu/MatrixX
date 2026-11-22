#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for UAS Report (Section A & F)
1. Strong Scaling vs Weak Scaling Curves (Amdahl's Law vs Gustafson's Law)
2. Communication Bottleneck & Latency Decomposition (Computation vs MPI Overhead)
"""

import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def generate_scaling_comparison():
    procs = [1, 2, 4, 8, 16]
    ideal_eff = [100.0] * 5
    weak_eff = [100.0, 96.2, 92.4, 88.5, 83.1]
    strong_eff = [100.0, 92.5, 86.3, 73.5, 56.8]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    ax.plot(procs, ideal_eff, 'k--', label='Ideal Linear Efficiency (100%)', linewidth=1.5, alpha=0.7)
    ax.plot(procs, weak_eff, 'o-', color='#0284c7', label="Weak Scaling (Gustafson's Law, Workload/Core Constant)", linewidth=2.5, markersize=8)
    ax.plot(procs, strong_eff, 's-', color='#e11d48', label="Strong Scaling (Amdahl's Law, N=4096 Constant)", linewidth=2.5, markersize=8)

    ax.set_title('Parallel Scalability: Strong Scaling vs Weak Scaling Efficiency', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Processors / Nodes (P)', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=11, fontweight='semibold')
    ax.set_xticks(procs)
    ax.set_ylim(40, 105)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower left')

    # Value annotations
    for x, y in zip(procs, weak_eff):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, color='#0284c7', fontweight='bold')
    for x, y in zip(procs, strong_eff):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=9, color='#e11d48', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "strong_vs_weak_scaling.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Generated: {out_path}")

def generate_bottleneck_breakdown():
    sizes = ['512×512', '1024×1024', '2048×2048', '4096×4096']
    compute_pct = np.array([72.2, 81.9, 88.8, 94.4])
    comm_pct = np.array([27.8, 18.1, 11.2, 5.6])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    x = np.arange(len(sizes))
    width = 0.55

    p1 = ax.bar(x, compute_pct, width, label='Computation Time (O(N^3))', color='#2563eb', edgecolor='white')
    p2 = ax.bar(x, comm_pct, width, bottom=compute_pct, label='MPI Communication & Latency Overhead (O(N^2))', color='#f43f5e', edgecolor='white')

    ax.set_title('Communication Bottleneck: Computation vs Communication Share (%)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Matrix Dimension (N × N)', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Percentage of Total Runtime (%)', fontsize=11, fontweight='semibold')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontweight='semibold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

    for i in range(len(sizes)):
        ax.text(i, compute_pct[i] / 2, f"{compute_pct[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax.text(i, compute_pct[i] + comm_pct[i] / 2, f"{comm_pct[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "communication_bottleneck.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Generated: {out_path}")

if __name__ == "__main__":
    generate_scaling_comparison()
    generate_bottleneck_breakdown()
