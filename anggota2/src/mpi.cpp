#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[]) {
    // 1. Inisialisasi lingkungan MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // 2. Tentukan pembagian baris matriks A untuk setiap proses (Dekomposisi Baris)
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

    // 3. Alokasi memori
    // Matriks A dan C hanya wajib dialokasikan penuh di Rank 0 (Master)
    vector<double> A;
    vector<double> C;
    if (rank == 0) {
        A.resize((size_t)N * N);
        C.resize((size_t)N * N, 0.0);

        // Inisialisasi matriks A sesuai standar Anggota 1
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[(size_t)i * N + j] = ((i + j) % 100) * 0.01;
            }
        }
    }

    // Matriks B dibutuhkan lengkap oleh semua proses
    vector<double> B((size_t)N * N);
    if (rank == 0) {
        // Inisialisasi matriks B sesuai standar Anggota 1
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B[(size_t)i * N + j] = ((i * 3 + j) % 100) * 0.01;
            }
        }
    }

    // Sub-matriks A dan C untuk masing-masing proses worker
    vector<double> A_local((size_t)my_rows * N);
    vector<double> C_local((size_t)my_rows * N, 0.0);

    double totalTime = 0.0;
    double minTime = 1e100;

    // 4. Pengujian Benchmark Berulang
    for (int r = 0; r < repeats; r++) {
        fill(C_local.begin(), C_local.end(), 0.0);
        if (rank == 0) {
            fill(C.begin(), C.end(), 0.0);
        }

        // Sinkronisasi seluruh proses sebelum mulai mengukur waktu
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        // 5. Distribusikan baris matriks A ke masing-masing proses
        MPI_Scatterv(
            A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
            A_local.data(), my_rows * N, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        // 6. Broadcast matriks B lengkap ke seluruh proses
        MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // 7. Komputasi lokal perkalian matriks (Loop i-k-j untuk optimalisasi cache)
        for (int i = 0; i < my_rows; i++) {
            for (int k = 0; k < N; k++) {
                double temp = A_local[(size_t)i * N + k];
                for (int j = 0; j < N; j++) {
                    C_local[(size_t)i * N + j] += temp * B[(size_t)k * N + j];
                }
            }
        }

        // 8. Kumpulkan sub-matriks hasil C dari semua proses kembali ke Rank 0
        MPI_Gatherv(
            C_local.data(), my_rows * N, MPI_DOUBLE,
            C.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        // Sinkronisasi setelah seluruh proses selesai mengirim hasil
        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();

        double elapsed = end - start;
        if (rank == 0) {
            totalTime += elapsed;
            if (elapsed < minTime) {
                minTime = elapsed;
            }
        }
    }

    // 9. Validasi hasil (Checksum) & Tampilkan Hasil di Rank 0
    if (rank == 0) {
        double checksum = 0.0;
        for (double value : C) {
            checksum += value;
        }

        cout << fixed << setprecision(9);
        cout << "Method: MPI\n";
        cout << "Matrix Size: " << N << " x " << N << "\n";
        cout << "Processes: " << size << "\n";
        cout << "Average Execution Time: " << (totalTime / repeats) << " seconds\n";
        cout << "Minimum Execution Time: " << minTime << " seconds\n";
        cout << "Checksum: " << checksum << "\n";
    }

    // 10. Selesai dan bersihkan lingkungan MPI
    MPI_Finalize();
    return 0;
}
