#ifndef NN_UTILS_CPP
#define NN_UTILS_CPP

#include "NN-utils.hpp"
#include "ConnectedLayer.hpp"

void arrayAdd(double* base, double* add, int length) {
    for (int i = 0; i < length; i++) {
        base[i] += add[i];
    }
} 

void arraySub(double* base, double* sub, int length) {
    for (int i = 0; i < length; i++) {
        base[i] -= sub[i];
    }
}

int compareActualToExpected(Matrix expected, double* actual)
{
    double* possible_choice = new double[expected.cols];
    zeroArray(possible_choice, expected.cols);
    int guess = 0;
    double min = 0.0;
    bool init = true;
    for (int i = 0; i < expected.rows; i++) {
        expected.getRow(i, possible_choice);
        double error = 0.0;
        for (int k = 0; k < expected.cols; k++) {
            error += abs(possible_choice[k] - actual[k]);
        }
        if (init) {
            min = error;
            init = false;
        }
        else {
            if (error < min) {
                min = error;
                guess = i;
            }
        }
    }
    delete[] possible_choice;
    return guess;
}

int convertBinToInt(double* array, int size)
{
    int total = 0;
    for (int i = 0; i < size; i++) {
        total += (int)(array[i] * (pow(2, size - 1 - i)));
    }
    return total;
}

void copyArray(double* toCopy, double* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = toCopy[i];
    }
}

//Matrix createKernel(int rows, int cols, int filter_type){
//    double* elements;
//
//    //Construct the kernels elements based on inputted filter
//    switch (filter_type) {
//    case BLUR: elements = new double[rows * cols];
//               copyArray(BLUR_KERNEL, elements, rows * cols);
//               break;
//    case GAUSSIAN_BLUR: elements = new double[rows * cols];
//                        copyArray(GAUSSIAN_BLUR_KERNEL, elements, rows * cols);
//                        break;
//    default: elements = new double[rows * cols];
//             copyArray(BLUR_KERNEL, elements, rows * cols);
//             break;
//
//    }
//    Matrix kernel(rows, cols, elements);
//    return kernel;
//}

// Formats and displays a 1-D array with as many rows/cols as given 
void displayArray(std::string msg, double* array, int rows, int cols, bool show_dim) {
    if (show_dim) {
        std::cout << msg << "(" << rows << ", " << cols << ")" << std::endl;
    }
    else {
        std::cout << msg;
    }
    std::cout << "[";
    for (int element = 1; element <= rows * cols; element++) {
        std::cout << array[element - 1];
        if (element % cols == 0) {
            std::cout << "]\n";
            if (element < rows * cols) {
                std::cout << "[";
            }
        }
        else {
            std::cout << ", ";
        }
    }
    if (rows == 0 || cols == 0) {
        std::cout << "]\n";
    }
}
void displayArray(std::string msg, int* array, int rows, int cols, bool show_dim) {
    if (show_dim) {
        std::cout << msg << "(" << rows << ", " << cols << ")" << std::endl;
    }
    else {
        std::cout << msg;
    }
    std::cout << "[";
    for (int element = 1; element <= rows * cols; element++) {
        std::cout << array[element - 1];
        if (element % cols == 0) {
            std::cout << "]\n";
            if (element < rows * cols) {
                std::cout << "[";
            }
        }
        else {
            std::cout << ", ";
        }
    }
    if (rows == 0 || cols == 0) {
        std::cout << "]\n";
    }
}

void dot(Matrix A, Matrix B, Matrix *result) {
    if (A.cols != B.rows) {
        return;
    }
    result->rows = A.rows;
    result->cols = B.cols;
    result->array = new double[A.rows * B.cols];
    result->deleteDynMemOnDestroy();

    // B is a scalar
    if (B.rows == 1 && B.cols == 1) {
        for (int i = 0; i < A.rows; i++) {
            result->array[i] = A.array[i] * B.array[0];
        }
    }
    // Matrix Multiplication
    else {
        for (int curr_row = 0; curr_row < A.rows; curr_row++) {
            for (int curr_col = 0; curr_col < B.cols; curr_col++) {
                double sum = 0.0;
                for (int k = 0; k < B.rows; k++) {
                    int first_index = k + (curr_row * A.cols);
                    int second_index = curr_col + (k * B.cols);

                    sum += A.array[first_index] * B.array[second_index];
                }
                int assign_index = curr_col + curr_row * B.cols;
                result->array[assign_index] = sum;
            }
        }
    }
}


void dot(double* A, double* B, double* result, int Arows, int Acols, int Brows, int Bcols) {
    // B is a scalar
    if (Brows == 1 && Bcols == 1) {
        for (int i = 0; i < Arows; i++) {
            result[i] = A[i] * B[0];
        }
    }
    // Matrix Multiplication
    else {
        for (int curr_row = 0; curr_row < Arows; curr_row++) {
            for (int curr_col = 0; curr_col < Bcols; curr_col++) {
                double sum = 0.0;
                for (int k = 0; k < Brows; k++) {
                    int first_index = k + (curr_row * Acols);
                    int second_index = curr_col + (k * Bcols);

                    sum += A[first_index] * B[second_index];
                }
                int assign_index = curr_col + curr_row * Bcols;
                result[assign_index] = sum;
            }
        }
    }
}

void generateRandArray(double* arr, int size) {
    // Primer Rand Call
    int ignore = rand();

    for (int i = 0; i < size; i++) {
        arr[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));//may be a good spot to try some computations using exp to compare to python
}// e^x as in euler's function not some exponent but sometimes 2^x is substituted?

double sigmoid_derivative(double x) {
    return x * (1 - x);
}

// Array Transposition
void transpose(double* array, double* transposed, int M, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            transposed[row + col * M] = array[row * N + col];
        }
    }
}

void zeroArray(double* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = 0.0;
    }
}

#endif