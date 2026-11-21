#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <mpi.h>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[]) {
    // 1. Inisialisasi MPI dengan dukungan thread (hanya thread utama memanggil MPI)
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            cerr << "MPI tidak mendukung MPI_THREAD_FUNNELED!\n";
        }
        MPI_Finalize();
        return 1;
    }

    int N = 512;
    int repeats = 5;

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) repeats = atoi(argv[2]);

    if (N <= 0 || repeats <= 0) {
        if (rank == 0) {
            cerr << "Ukuran matriks dan jumlah pengulangan harus bernilai positif!\n";
        }
        MPI_Finalize();
        return 1;
    }

    // 2. Atur OpenMP: jumlah thread tetap (dari OMP_NUM_THREADS), catat jumlah aktual
    omp_set_dynamic(0);
    int actualThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actualThreads = omp_get_num_threads();
    }

    // 3. Pembagian baris matriks A antarproses (dekomposisi baris)
    int base_rows = N / size;
    int rem_rows = N % size;

    vector<int> sendcounts(size);
    vector<int> displs(size);
    int offset = 0;

    for (int p = 0; p < size; p++) {
        int rows_p = base_rows + (p < rem_rows ? 1 : 0);
        sendcounts[p] = rows_p * N;
        displs[p] = offset * N;
        offset += rows_p;
    }

    int my_rows = base_rows + (rank < rem_rows ? 1 : 0);

    // 4. Alokasi memori: A dan C penuh hanya di rank 0, B penuh di semua proses
    vector<double> A;
    vector<double> C;
    vector<double> B((size_t)N * N);

    if (rank == 0) {
        A.resize((size_t)N * N);
        C.resize((size_t)N * N, 0.0);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[(size_t)i * N + j] = ((i + j) % 100) * 0.01;
                B[(size_t)i * N + j] = ((i * 3 + j) % 100) * 0.01;
            }
        }
    }

    vector<double> A_local((size_t)my_rows * N);
    vector<double> C_local((size_t)my_rows * N, 0.0);

    double totalTime = 0.0;
    double minTime = 1e100;
    double totalCompute = 0.0;

    // 5. Benchmark berulang
    for (int r = 0; r < repeats; r++) {
        fill(C_local.begin(), C_local.end(), 0.0);
        if (rank == 0) {
            fill(C.begin(), C.end(), 0.0);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        // Distribusi baris A dan broadcast B
        MPI_Scatterv(
            A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
            A_local.data(), my_rows * N, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Komputasi lokal: baris dibagi ke thread OpenMP.
        // Tiap thread menulis baris C_local yang berbeda, jadi tidak ada race condition.
        double computeStart = MPI_Wtime();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < my_rows; i++) {
            for (int k = 0; k < N; k++) {
                double temp = A_local[(size_t)i * N + k];
                for (int j = 0; j < N; j++) {
                    C_local[(size_t)i * N + j] += temp * B[(size_t)k * N + j];
                }
            }
        }

        double computeEnd = MPI_Wtime();

        // Kumpulkan hasil ke rank 0
        MPI_Gatherv(
            C_local.data(), my_rows * N, MPI_DOUBLE,
            C.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();

        // Waktu komputasi terlama antarproses (dihitung di luar timer total)
        double localCompute = computeEnd - computeStart;
        double maxCompute = 0.0;
        MPI_Reduce(&localCompute, &maxCompute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double elapsed = end - start;
            totalTime += elapsed;
            totalCompute += maxCompute;
            if (elapsed < minTime) {
                minTime = elapsed;
            }
        }
    }

    // 6. Validasi (checksum) dan output di rank 0
    if (rank == 0) {
        double checksum = 0.0;
        for (double value : C) {
            checksum += value;
        }

        cout << fixed << setprecision(9);
        cout << "Method: Hybrid\n";
        cout << "Matrix Size: " << N << " x " << N << "\n";
        cout << "Processes: " << size << "\n";
        cout << "Threads per Process: " << actualThreads << "\n";
        cout << "Total Threads: " << size * actualThreads << "\n";
        cout << "Average Execution Time: " << (totalTime / repeats) << " seconds\n";
        cout << "Minimum Execution Time: " << minTime << " seconds\n";
        cout << "Average Compute Time: " << (totalCompute / repeats) << " seconds\n";
        cout << "Checksum: " << checksum << "\n";
    }

    MPI_Finalize();
    return 0;
}
