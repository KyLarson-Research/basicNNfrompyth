#ifndef CONNECTED_LAYER_HPP
#define CONNECTED_LAYER_HPP
#include <iostream>
#include "Layer.hpp"
#include "NN-utils.hpp"
#include <fstream>

// Every input neuron is connected to every output neuron
class ConnectedLayer : public Layer {
	public:
		int input_cnt = 0;
		int output_cnt = 0;
		double* weights = nullptr;
		double* bias = nullptr;
	private:
		
		double* inputs = nullptr;
		double* outputs = nullptr;
		double* weights_error = nullptr;
		double* back_propagation_result = nullptr;

	public:
		// Constructors
		ConnectedLayer() {}

		ConnectedLayer(int input_cnt, int output_cnt);

		//Update weights and biases by subtracting the derivative of the error
		void backwardPropagation(double* output_err, double learning_rate);

		// Delete any members that used the new keyword
		void clear();

		// Formats object's members and displays them
		void display();

		// Calculate layer output using input
		void forwardPropagation(double* inputs);

		double* getBPR();
		double* getOutputs();

		//Save this object to a file
		void save(std::ofstream *fh);
		
		void setupArrays(bool setRandomVals);

		//Set weights/bias, rather than random values
		void setWeights(double* weights);

		void setBias(double* bias);
};

#endif