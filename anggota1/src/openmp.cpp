
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {

    int N = 512;
    int repeats = 5;

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) repeats = atoi(argv[2]);

    if (N <= 0 || repeats <= 0) {
        cerr << "Input harus positif!\n";
        return 1;
    }

    vector<double> A((size_t)N * N);
    vector<double> B((size_t)N * N);
    vector<double> C((size_t)N * N, 0.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[(size_t)i * N + j] = ((i + j) % 100) * 0.01;
            B[(size_t)i * N + j] = ((i * 3 + j) % 100) * 0.01;
        }
    }

    omp_set_dynamic(0);

    double totalTime = 0.0;
    double minTime = 1e100;
    int actualThreads = 1;

    for (int r = 0; r < repeats; r++) {

        fill(C.begin(), C.end(), 0.0);

        auto start = steady_clock::now();

        #pragma omp parallel
        {
            #pragma omp single
            actualThreads = omp_get_num_threads();

            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {

                for (int k = 0; k < N; k++) {

                    double temp = A[(size_t)i * N + k];

                    for (int j = 0; j < N; j++) {
                        C[(size_t)i * N + j] +=
                            temp * B[(size_t)k * N + j];
                    }
                }
            }
        }

        auto end = steady_clock::now();

        double elapsed = duration<double>(end - start).count();

        totalTime += elapsed;

        if (elapsed < minTime) {
            minTime = elapsed;
        }
    }

    double checksum = 0.0;

    for (double value : C) {
        checksum += value;
    }

    cout << fixed << setprecision(9);

    cout << "Method: OpenMP\n";
    cout << "Matrix Size: " << N << " x " << N << "\n";
    cout << "Threads: " << actualThreads << "\n";

    cout << "Average Execution Time: "
         << totalTime / repeats << " seconds\n";

    cout << "Minimum Execution Time: "
         << minTime << " seconds\n";

    cout << "Checksum: " << checksum << "\n";

    return 0;
}

