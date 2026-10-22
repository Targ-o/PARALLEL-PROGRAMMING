#include <omp.h>
import std;
using namespace std;

vector<vector<double>> readMatrix(const string& filename) {
    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("file can't be opened! file: " + filename);
    }

    int n;
    file >> n;

    vector<vector<double>> matrix(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file >> matrix[i][j];
        }
    }

    return matrix;
}

vector<vector<double>> umnozenie(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    int threads
) {
    int n = A.size();

    vector<vector<double>> C(n, vector<double>(n, 0.0));

#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double aik = A[i][k];
            for (int j = 0; j < n; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }

    return C;
}

void writeMatrix(
    const string& filename,
    const vector<vector<double>>& matrix
) {
    ofstream file(filename);

    int n = matrix.size();

    file << n << "\n";

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i][j];

            if (j + 1 < n) {
                file << " ";
            }
        }

        file << "\n";
    }
}

int main() {

    try {
        vector<int> sizes = { 20, 200, 400, 800, 1200, 1600, 2000 };
        vector<int> threadList = { 1, 2, 4, 8 };

        println("Cores available: {}", omp_get_num_procs());
        println("");

        for (int n : sizes) {

            auto A = readMatrix("data/matrix_a_" + to_string(n) + ".txt");
            auto B = readMatrix("data/matrix_b_" + to_string(n) + ".txt");

            println("Matrix size: {} x {}", n, n);

            double time1 = 0.0;
            vector<vector<double>> C;

            for (int threads : threadList) {

                double start = omp_get_wtime();

                C = umnozenie(A, B, threads);

                double time = omp_get_wtime() - start;

                if (threads == 1) {
                    time1 = time;
                }

                println("  threads: {} | time: {:.4f} s | speedup: {:.2f}",
                    threads, time, time1 / time);
            }

            writeMatrix("results/matrix_C_" + to_string(n) + ".txt", C);
            println("");
        }

        println("operation: done!!");
    }
    catch (const exception& e) {
        cerr << "Err: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
