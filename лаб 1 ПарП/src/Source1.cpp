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
    const vector<vector<double>>& B
) {
    int n = A.size();


        vector<vector<double>> C(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
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

            vector<int> sizes = {
                20, 200, 400, 800, 1200, 1600, 2000
            };

            for (int n : sizes) {

                string fileA = "data/matrix_a_" + to_string(n) + ".txt";
                string fileB = "data/matrix_b_" + to_string(n) + ".txt";

                auto A = readMatrix(fileA);
                auto B = readMatrix(fileB);

                if (A.size() != B.size()) {
                    throw runtime_error("matrix sizes are different! canot proceed!");
                }

                println("Matrix size: {} x {}", n, n);
                println("Elements count: {}", static_cast<long long>(n) * n);
                println("Task volume: {}", static_cast<long long>(n) * n * n);

                auto start = chrono::high_resolution_clock::now();

                auto C = umnozenie(A, B);

                auto finish = chrono::high_resolution_clock::now();

                chrono::duration<double> time = finish - start;

                string resultFile =
                    "results/matrix_C_" + to_string(n) + ".txt";

                writeMatrix(resultFile, C);

                println("Execution time: {} seconds", time.count());
                println("Result: {}", resultFile);
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
