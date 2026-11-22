#!/usr/bin/env bash
# ==============================================================================
# MatrixX HPC Master Execution & Validation Suite
# Compiles and runs Sequential, OpenMP, MPI, and Hybrid Matrix Multiplication
# ==============================================================================

set -e

MATRIX_SIZE=${1:-2048}
REPEATS=${2:-5}

echo "========================================================================"
echo "           MATRIXX HPC BENCHMARK & VALIDATION SUITE                     "
echo "           Target Matrix: ${MATRIX_SIZE}x${MATRIX_SIZE} | Trials: ${REPEATS}                        "
echo "========================================================================"

echo ""
echo "[1/4] Compiling and Running Sequential (Anggota 1)..."
g++ -O3 -march=native -std=c++17 anggota1/src/sequential.cpp -o anggota1/sequential
./anggota1/sequential $MATRIX_SIZE $REPEATS

echo ""
echo "[2/4] Compiling and Running OpenMP (Anggota 1, 4 Threads)..."
g++ -O3 -march=native -std=c++17 -fopenmp anggota1/src/openmp.cpp -o anggota1/openmp
OMP_NUM_THREADS=4 ./anggota1/openmp $MATRIX_SIZE $REPEATS

if command -v mpicxx &> /dev/null; then
    echo ""
    echo "[3/4] Compiling and Running MPI Distributed (Anggota 2, 4 Ranks)..."
    mpicxx -O3 -march=native -std=c++17 anggota2/src/mpi.cpp -o anggota2/mpi
    mpirun --oversubscribe -np 4 ./anggota2/mpi $MATRIX_SIZE $REPEATS

    echo ""
    echo "[4/4] Compiling and Running Hybrid MPI+OpenMP (Anggota 3, 2 Ranks x 2 Threads)..."
    mpicxx -O3 -march=native -std=c++17 -fopenmp anggota3/src/hybrid.cpp -o anggota3/hybrid
    mpirun --oversubscribe -np 2 ./anggota3/hybrid $MATRIX_SIZE $REPEATS 2
else
    echo ""
    echo "[NOTICE] mpicxx not found in current environment. Skipping MPI and Hybrid binary execution."
    echo "         To run MPI and Hybrid, execute inside WSL2 Ubuntu or Docker container."
fi

echo ""
echo "========================================================================"
echo " [SUCCESS] Benchmark pipeline finished. Results logged to CSVs.        "
echo "========================================================================"
