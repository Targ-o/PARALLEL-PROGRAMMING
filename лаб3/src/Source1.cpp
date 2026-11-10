
#define OMPI_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#include <mpi.h>
 
import std;
using namespace std;
 
vector<double> readMatrix(const string& filename, int& n) {
    ifstream file(filename);
 
    if (!file.is_open()) {
        throw runtime_error("file can't be opened! file: " + filename);
    }
 
    file >> n;
 
    vector<double> matrix(static_cast<size_t>(n) * n);
 
    for (size_t i = 0; i < matrix.size(); i++) {
        file >> matrix[i];
    }
 
    return matrix;
}
 

void umnozenie(
    const double* A_part,
    const double* B,
    double* C_part,
    int rows,
    int n
) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A_part[static_cast<size_t>(i) * n + k] *
                       B[static_cast<size_t>(k) * n + j];
            }
            C_part[static_cast<size_t>(i) * n + j] = sum;
        }
    }
}
 
void writeMatrix(
    const string& filename,
    const vector<double>& matrix,
    int n
) {
    ofstream file(filename);
 
    if (!file.is_open()) {
        throw runtime_error("file can't be opened for writing! file: " + filename);
    }
 
    file << n << "\n";
 
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[static_cast<size_t>(i) * n + j];
 
            if (j + 1 < n) {
                file << " ";
            }
        }
 
        file << "\n";
    }
}
 
int main(int argc, char** argv) {
 
    MPI_Init(&argc, &argv);
 
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    try {
 
        vector<int> sizes = {
            20, 200, 400, 800, 1200, 1600, 2000
        };
 
        if (rank == 0) {
            println("MPI processes: {}", size);
            println("");
        }
 
        for (int n : sizes) {
 
            vector<double> A, B, C;
 
            if (rank == 0) {
                string fileA = "data/matrix_a_" + to_string(n) + ".txt";
                string fileB = "data/matrix_b_" + to_string(n) + ".txt";
 
                int nA = 0, nB = 0;
                A = readMatrix(fileA, nA);
                B = readMatrix(fileB, nB);
 
                if (nA != nB || nA != n) {
                    throw runtime_error("matrix sizes are different! canot proceed!");
                }
 
                C.resize(static_cast<size_t>(n) * n);
            }
            else {
                B.resize(static_cast<size_t>(n) * n);
            }

            vector<int> sendCounts(size), displs(size);
            int base = n / size;
            int rest = n % size;
            int offset = 0;
 
            for (int p = 0; p < size; p++) {
                int rowsP = base + (p < rest ? 1 : 0);
                sendCounts[p] = rowsP * n;
                displs[p] = offset;
                offset += sendCounts[p];
            }
 
            int myRows = sendCounts[rank] / n;
            vector<double> A_part(static_cast<size_t>(myRows) * n);
            vector<double> C_part(static_cast<size_t>(myRows) * n);
 
            MPI_Barrier(MPI_COMM_WORLD);
            double tStart = MPI_Wtime();
 
            MPI_Scatterv(
                A.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                A_part.data(), sendCounts[rank], MPI_DOUBLE,
                0, MPI_COMM_WORLD
            );
 

            MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 
            double tCompStart = MPI_Wtime();
            umnozenie(A_part.data(), B.data(), C_part.data(), myRows, n);
            double tCompLocal = MPI_Wtime() - tCompStart;

            MPI_Gatherv(
                C_part.data(), sendCounts[rank], MPI_DOUBLE,
                C.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD
            );
 
            double tTotalLocal = MPI_Wtime() - tStart;
 
            double tTotal = 0.0, tComp = 0.0;
            MPI_Reduce(&tTotalLocal, &tTotal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&tCompLocal, &tComp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
            if (rank == 0) {
                string resultFile =
                    "results/matrix_C_" + to_string(n) + ".txt";
 
                writeMatrix(resultFile, C, n);
 
                println("Matrix size: {} x {}", n, n);
                println("Elements count: {}", static_cast<long long>(n) * n);
                println("Task volume: {}", static_cast<long long>(n) * n * n);
                println("Processes: {}", size);
                println("Computation time: {} seconds", tComp);
                println("Total time (with communication): {} seconds", tTotal);
                println("Result: {}", resultFile);
                println("");
            }
        }
 
        if (rank == 0) {
            println("operation: done!!");
        }
 
    }
    catch (const exception& e) {
        cerr << "Err (rank " << rank << "): " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1); 
        return 1;
    }
 
    MPI_Finalize();
    return 0;
}
 