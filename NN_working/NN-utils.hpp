#ifndef NN_UTILS_HPP
#define NN_UTILS_HPP

#include <iostream>
#include <random>
#include "Layer.hpp"

const bool DELETE_MEM_ON_DESTROY = true;
const bool DO_NOT_USE_DELETE = false;

enum filter_types{BLUR, GAUSSIAN_BLUR};

static double BLUR_KERNEL[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
static double GAUSSIAN_BLUR_KERNEL[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};

void displayArray(std::string msg, double* array, int rows, int cols, bool show_dim = true);
void displayArray(std::string msg, int* array, int rows, int cols, bool show_dim = true);

// Object for handling 2D arrays
class Matrix{
public:
	int rows = 0;
	int cols = 0;
	double* array = nullptr;
private:
	bool arrayCreatedWithNew = false;

public:
	Matrix() {}

	Matrix(int rows, int cols, double* array, bool arrayCreatedWithNew=false) {
		this->rows = rows;
		this->cols = cols;
		this->array = array;
		this->arrayCreatedWithNew = arrayCreatedWithNew;
	}

	//Destructor, automatically release memory if it has been dynamically allocated
	~Matrix() {
		if (arrayCreatedWithNew && array != nullptr) {
			delete[] array;
		}
	}

	void display(std::string msg) {
		if (array != nullptr) {
			displayArray(msg, array, rows, cols);
		}
		else {
			std::cout << "\n[Matrix.display] double* array is NULL\n";
		}
	}

	double get(int index) {
		return array[index];
	}

	void getRow(int row, double* output) {
		for (int i = 0; i < cols; i++) {
			output[i] = array[i + (cols * row)];
		}
	}

	void deleteDynMemOnDestroy() {
		arrayCreatedWithNew = true;
	}
};

// Function Prototypes
void arrayAdd(double* base, double* add, int length);

void arraySub(double* base, double* sub, int length);

int compareActualToExpected(Matrix expected, double* actual);

int convertBinToInt(double* array, int size);

void copyArray(double* toCopy, double* dst, int size);

//Matrix createKernel(int width, int height, int filter_type);

void dot(Matrix A, Matrix B, Matrix *result);

void dot(double* arrayOne, double* arrayTwo, double* result, 
	     int Arows, int Acols, int Brows, int Bcols);

void generateRandArray(double* arr, int size);

double sigmoid(double x);

double sigmoid_derivative(double x);

void transpose(double* array, double* transposed, int M, int N);

void zeroArray(double* array, int size);
#endif