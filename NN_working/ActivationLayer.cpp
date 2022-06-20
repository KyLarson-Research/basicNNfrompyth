#ifndef ACTIVATION_LAYER_CPP
#define ACTIVATION_LAYER_CPP

#include "ActivationLayer.hpp"

//Constructors
ActivationLayer::ActivationLayer(int input_cnt, std::string activateName, std::string activatePrimeName) {

	// Setup members with constructor arguments
	this->activateName = activateName;
	this->activatePrimeName = activatePrimeName;

	this->activate = getActivationFromStr(activateName);
	this->activate_prime = getActivationFromStr(activatePrimeName);

	this->input_cnt = input_cnt;

	input = new double[input_cnt];
	outputs = new double[input_cnt];
	outputs_prime = new double[input_cnt];
	back_propagation_result = new double[input_cnt];
	
	zeroArray(input, input_cnt);
	zeroArray(outputs, input_cnt);
	zeroArray(outputs_prime, input_cnt);
	zeroArray(back_propagation_result, input_cnt);
}

void ActivationLayer::backwardPropagation(double* output_err, double learning_rate) {
	if (activate_prime == nullptr) {
		return;
	}
	// Apply activate_prime to the input
	for (int i = 0; i < input_cnt; i++) {
		back_propagation_result[i] = activate_prime(input[i]) * output_err[i];
	}
}

void ActivationLayer::clear() {
	if (input != nullptr) { delete[] input; }
	if (outputs != nullptr) { delete[] outputs; }
	if (outputs_prime != nullptr) { delete[] outputs_prime; }
	if (back_propagation_result != nullptr) { delete[] back_propagation_result; }
}

void ActivationLayer::display() {
	std::cout << "Activation Layer: " << ID << std::endl;
	std::cout << "Activate: " << activateName << std::endl 
			  << "Activate Prime: " << activatePrimeName << std::endl;
	displayArray("Input: ", input, 1, input_cnt);
	displayArray("Output: ", outputs, 1, input_cnt);
	displayArray("Output Prime: ", outputs_prime, 1, input_cnt);
}

//Calculate activated input
void ActivationLayer::forwardPropagation(double* input) {
	if (activate == nullptr) {
		return;
	}
	// Apply activation function to every input
	for (int i = 0; i < input_cnt; i++) {
		this->input[i] = input[i];
		outputs[i] = activate(input[i]);
	}
}

double* ActivationLayer::getOutputs(){
	return outputs;
}

double* ActivationLayer::getBPR(){
	return back_propagation_result;
}

void ActivationLayer::save(std::ofstream *fh){
	if (!(*fh).is_open()) {
		std::cout << "[ActivationLayer::save] File Handle Not Open, Error Saving Object\n";
		return;
	}

	//Setup header
	*fh << "Activation_Layer";
	//Save this object's data
	*fh << "\nInput/Output " << input_cnt;
	*fh << "\nActivate " << activateName;
	*fh << "\nActivate_Prime " << activatePrimeName;
}

#endif
