#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>

// Super class for layer object
class Layer {
	public:
		int ID = 0;
	public:
		void forwardPropagation(double *input) {
			//
		}
		void backwardPropagation(double* out_err, double learning_rate) {
			//
		}

		void save(){}

		//Setters
		void setID(int ID) { this->ID = ID; }
};

#endif
