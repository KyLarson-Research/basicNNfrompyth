#ifndef ERROR_HPP
#define ERROR_HPP

#include <iostream>

inline double meanSquaredError(double expected, double *predicted, int data_points) {
	double total = 0.0;
	for (int i = 0; i < data_points; i++) {
		double temp = expected - predicted[i];
		temp *= temp;
		total += temp;
	}
	return total / data_points;
}

inline void meanSquaredErrorPrime(double* expected, double* predicted, double* result, int data_points) {
	for (int i = 0; i < data_points; i++) {
		result[i] = (2 * (predicted[i] - expected[i])) / data_points;
	}
}

inline auto getErrorFromStr(std::string name) {
	if (name == "meanSquaredError") {
		return &meanSquaredError;
	}
	else {
		return &meanSquaredError;
	}
}

inline auto getErrorPrimeFromStr(std::string name) {
	if (name == "meanSquaredErrorPrime") {
		return &meanSquaredErrorPrime;
	}
	else {
		return &meanSquaredErrorPrime;
	}
}

#endif