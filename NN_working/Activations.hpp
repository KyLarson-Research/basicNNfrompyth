#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <iostream>

// Activation function definitions
inline double tan_hb(double input) {
	return tanh(input);
}

inline double tan_hb_prime(double input) {
	double temp = tanh(input);
	return 1 - (temp * temp);
}

inline auto getActivationFromStr(std::string name) {
	if (name == "tan_hb") {
		return &tan_hb;
	}
	else {
		return &tan_hb_prime;
	}
}

#endif