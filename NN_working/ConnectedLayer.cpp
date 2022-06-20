#ifndef CONNECTED_LAYER_CPP
#define CONNECTED_LAYER_CPP

#include "ConnectedLayer.hpp"

// Constructor
ConnectedLayer::ConnectedLayer(int input_cnt, int output_cnt) {

	// Setup members with constructor arguments
	this->input_cnt = input_cnt;
	this->output_cnt = output_cnt;

	// Create random arrays for weights and biases
	weights = new double[input_cnt*output_cnt];
	generateRandArray(weights, input_cnt * output_cnt);

	bias = new double[output_cnt];
	generateRandArray(bias, output_cnt);

	// Zero out input, output, input error, and weight error arrays
	inputs = new double[input_cnt];
	outputs = new double[output_cnt];
	back_propagation_result = new double[input_cnt];
	weights_error = new double[input_cnt * output_cnt];
	zeroArray(inputs, input_cnt);
	zeroArray(outputs, output_cnt);
	zeroArray(back_propagation_result, input_cnt);
	zeroArray(weights_error, input_cnt * output_cnt);

}

void ConnectedLayer::backwardPropagation(double* output_error, double learning_rate){
	// Transpose input and weight matrices
	double* inputs_transposed = new double[input_cnt];
	double* weights_transposed = new double[input_cnt * output_cnt];
	transpose(inputs, inputs_transposed, 1, input_cnt);
	transpose(weights, weights_transposed, input_cnt, output_cnt);
	
	//Calculate result of back propagation to feed to previous layer
	dot(output_error, weights_transposed, back_propagation_result, 1, output_cnt, output_cnt, input_cnt);

	//Calculate the error for the weights
	dot(inputs_transposed, output_error, weights_error, input_cnt, 1, 1, output_cnt);

	// Multiply errors by the learning rate
	for (int i = 0; i < input_cnt * output_cnt; i++) {
		weights_error[i] *= learning_rate;
	}
	for (int i = 0; i < output_cnt; i++) {
		output_error[i] *= learning_rate;
	}
	
	// Update weights and biases
	arraySub(weights, weights_error, input_cnt * output_cnt);
	arraySub(bias, output_error, output_cnt);

	//Clear allocated memory
	delete[] inputs_transposed;
	delete[] weights_transposed;
}

void ConnectedLayer::clear() {
	delete[] weights;
	delete[] bias;
	delete[] inputs;
	delete[] outputs;
	delete[] back_propagation_result;
	delete[] weights_error;
}

// Formats object's members and displays them
void ConnectedLayer::display() {
	std::cout << "\n===================\n";
	std::cout << "Connected Layer ID: " << ID << "\n";
	std::cout << "  Inputs:  " << input_cnt
	   << "\n  Outputs: " << output_cnt << "\n\n";
	displayArray("Weights: ", weights, input_cnt, output_cnt);
	displayArray("\nBias: ", bias, 1, output_cnt);
	displayArray("\nInputs: ", inputs, 1, input_cnt);
	displayArray("\nOutputs: ", outputs, 1, output_cnt);
	std::cout << "===================\n\n";
}

void ConnectedLayer::forwardPropagation(double* inputs) {
	for (int i = 0; i < input_cnt; i++) {
		this->inputs[i] = inputs[i];
	}

	//Calculate given output as the dot product of the inputs and weights + bias
	dot(inputs, weights, outputs, 1, input_cnt, input_cnt, output_cnt);
	arrayAdd(outputs, bias, output_cnt);

}
double* ConnectedLayer::getBPR(){
	return back_propagation_result;
}
double* ConnectedLayer::getOutputs(){
	return outputs;
}
void ConnectedLayer::save(std::ofstream *fh){
	if (!(*fh).is_open()) {
		std::cout << "[ConnectedLayer::save] File Handle Not Open, Error Saving Object\n";
		return;
	}

	//Setup header
	*fh << "Connected_Layer\n";

	//Save this object's data
	*fh << "Input " << input_cnt;
	*fh << "\nOutput " << output_cnt;
	*fh << "\nWeights ";
	for (int i = 0; i < input_cnt * output_cnt; i++) {
		*fh << weights[i];
		if (i < input_cnt * output_cnt - 1) {
			*fh << " ";
		}
	}
	*fh << "\nBias ";
	for (int i = 0; i < output_cnt; i++) {
		*fh << bias[i];
		if (i < output_cnt - 1) {
			*fh << " ";
		}
	}
}
void ConnectedLayer::setupArrays(bool setRandomVals){
	if (input_cnt == 0 || output_cnt == 0) {
		return;
	}
	// Create random arrays for weights and biases
	weights = new double[input_cnt * output_cnt];
	bias = new double[output_cnt];

	if (setRandomVals) {
		generateRandArray(weights, input_cnt * output_cnt);
		generateRandArray(bias, output_cnt);
	}

	// Zero out input, output, input error, and weight error arrays
	inputs = new double[input_cnt];
	outputs = new double[output_cnt];
	back_propagation_result = new double[input_cnt];
	weights_error = new double[input_cnt * output_cnt];
	zeroArray(inputs, input_cnt);
	zeroArray(outputs, output_cnt);
	zeroArray(back_propagation_result, input_cnt);
	zeroArray(weights_error, input_cnt * output_cnt);
}

void ConnectedLayer::setWeights(double* weights){
	for (int i = 0; i < input_cnt * output_cnt; i++) {
		this->weights[i] = weights[i];
	}
}

void ConnectedLayer::setBias(double* bias){
	for (int i = 0; i < output_cnt; i++) {
		this->bias[i] = bias[i];
	}
}
#endif