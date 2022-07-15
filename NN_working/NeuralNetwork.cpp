#ifndef NEURAL_NETWORK_CPP
#define NEURAL_NETWORK_CPP

#include "NeuralNetwork.hpp"

////////////////CONSTRUCTORS////////////////
NeuralNetwork::NeuralNetwork(std::string filename){
	if(load(filename) == FAILURE){
		std::cout << "[NeuralNetwork::NeuralNetwork(string)] Error Loading Network\n";
		layer_cnt = 0;
		clear();
	}
}

NeuralNetwork::NeuralNetwork(int ID, std::string errorName, std::string errorPrimeName){
	this->ID = ID;
	this->errorName = errorName;
	this->errorPrimeName = errorPrimeName;
	setError(getErrorFromStr(errorName), getErrorPrimeFromStr(errorPrimeName));
}

////////////////PUBLIC FUNCTIONS////////////////

void NeuralNetwork::addConnectedLayer(ConnectedLayer conn_layer) {
	//Create new connectedlist node to hold new layer
	ConnectedList *newNode = new ConnectedList;
	newNode->layer = conn_layer;
	newNode->next = nullptr;
	newNode->prev = nullptr;

	//Add the node to the list of connected layers
	ConnectedList** current = &conn_list;
	// Linked List Traversal
	while (*current != nullptr) {
		newNode->prev = *current;
		current = &((*current)->next);
	}
	*current = newNode;
	layer_cnt++;
	(*current)->layer.setID(layer_cnt);

	// Save the size of input/output for this network
	if (initial_input_cnt == 0) {
		initial_input_cnt = conn_layer.input_cnt;
	}
	final_output_cnt = conn_layer.output_cnt;
}

void NeuralNetwork::addActivationLayer(ActivationLayer act_layer) {

	//Create new activationlist node to hold new layer
	ActivationList* newNode = new ActivationList;
	newNode->layer = act_layer;
	newNode->next = nullptr;
	newNode->prev = nullptr;

	//Add the node to the list of connected layers
	ActivationList** current = &act_list;
	// Linked List Traversal
	while (*current != nullptr) {
		newNode->prev = *current;
		current = &((*current)->next);
	}
	*current = newNode;
	layer_cnt++;
	(*current)->layer.setID(layer_cnt);

	// Save the size of input/output for this network
	if (initial_input_cnt == 0) {
		initial_input_cnt = act_layer.input_cnt;
	}
	final_output_cnt = act_layer.input_cnt; // Activation layer |input| = |output|
}

void NeuralNetwork::clear() {
	conn_list = clearConnectedList(conn_list);
	act_list = clearActivationList(act_list);
}

void NeuralNetwork::display(bool show_detail) {
	std::cout << "Neural Network, ID: " << ID << std::endl;
	std::cout << "  # of Layers: " << layer_cnt << std::endl;
	std::cout << "  # of Inputs: " << initial_input_cnt << std::endl;
	std::cout << "  # of Outputs: " << final_output_cnt << std::endl;
	std::cout << "  ----------------\n";
	ConnectedList* temp_conn = conn_list;
	ActivationList* temp_act = act_list;
	int layer_index = 1;
	while (layer_index <= layer_cnt) {
		if (temp_conn != nullptr && temp_conn->layer.ID == layer_index) {
			if (show_detail) {
				temp_conn->layer.display();
			}
			else {
				std::cout << "  - Connected Layer, ID: " << temp_conn->layer.ID << std::endl;
			}
			temp_conn = temp_conn->next;
		}
		else if(temp_act != nullptr && temp_act->layer.ID == layer_index) {
			if (show_detail) {
				temp_act->layer.display();
			}
			else {
				std::cout << "  - Activation Layer, ID: " << temp_act->layer.ID << std::endl;
			}
			temp_act = temp_act->next;
		}
		else {
			std::cout << "\n[NeuralNetwork::display] Error: Layer ID Missing" << layer_index << std::endl;
		}
		layer_index++;
	}
}

void NeuralNetwork::predict(double* input, double* dst, int input_size){
	// Pointers for traversal of Network's layers
	ConnectedList* temp_conn = conn_list;
	ActivationList* temp_act = act_list;

	int output_size = input_size;
	double* working_output = new double[output_size];
	copyArray(input, working_output, input_size);
	for (int curr_layer = 1; curr_layer <= layer_cnt; curr_layer++) {
		if (temp_conn != nullptr && temp_conn->layer.ID == curr_layer) {
			temp_conn->layer.forwardPropagation(working_output);

			delete[] working_output;
			output_size = temp_conn->layer.output_cnt;
			working_output = new double[output_size];
			copyArray(temp_conn->layer.getOutputs(), working_output, output_size);

			// If next is null, leave ptr at tail to traverse back to head for back propagation
			if (temp_conn->next != nullptr) {
				temp_conn = temp_conn->next;
			}

		}
		else {
			temp_act->layer.forwardPropagation(working_output);

			delete[] working_output;
			output_size = temp_act->layer.input_cnt;
			working_output = new double[output_size];
			copyArray(temp_act->layer.getOutputs(), working_output, output_size);

			// If next is null, leave ptr at tail to traverse back to head for back propagation
			if (temp_act->next != nullptr) {
				temp_act = temp_act->next;
			}
		}
	}
	copyArray(working_output, dst, output_size);
	delete[] working_output;
}

void NeuralNetwork::setError(double(*error)(double, double*, int), 
	                        void(*error_prime)(double*, double*, double*, int)){
	this->error = error;
	this->error_prime = error_prime;
}

void NeuralNetwork::shutdown(std::string filename){
	save(filename);
	clear();
}

bool NeuralNetwork::isEmpty(){
	return layer_cnt == 0;
}

void NeuralNetwork::train(Matrix inputs, Matrix expected, int *expected_indexing, int epochs, double learning_rate){
	if (conn_list == nullptr && act_list == nullptr) {
		return;
	}
	
	std::cout << "\nBegin Training Section: \n";
	std::cout << "-----------------------\n\n";

	// Determine sample size
	int sample_cnt = inputs.rows;
	
	double* expected_sample = new double[expected.cols];
	double display_error = 0.0, error_delta = 0.0, prev_error = 0.0;
	bool init_error_delta = true;
	int quit_level = 5;
	double tolerance = 0.00001;
	int output_size = 0;

	// Pointers for layer LL traversal
	ConnectedList* temp_conn = conn_list;
	ActivationList* temp_act = act_list;

	// Iterate for number of epochs
	for (int cnt = 1; cnt <= epochs; cnt++) {
		display_error = 0.0;

		// Get start time
		auto epoch_start = std::chrono::system_clock::now();

		// Input each sample from inputs
		for (int test = 1; test <= sample_cnt; test++) {
			output_size = inputs.cols;
			double* working_output = new double[output_size];

			inputs.getRow(test - 1, working_output);

			temp_conn = conn_list;
			temp_act = act_list;

			// Forward propagate through all layers in the network
			for (int curr_layer = 1; curr_layer <= layer_cnt; curr_layer++) {
				if (temp_conn != nullptr && temp_conn->layer.ID == curr_layer) {
					temp_conn->layer.forwardPropagation(working_output);

					delete[] working_output;
					output_size = temp_conn->layer.output_cnt;
					working_output = new double[output_size];
					copyArray(temp_conn->layer.getOutputs(), working_output, output_size);

					// If next is null, leave ptr at tail to traverse back to head for back propagation
					if (temp_conn->next != nullptr) {
						temp_conn = temp_conn->next;
					}

				}
				else {
					temp_act->layer.forwardPropagation(working_output);

					delete[] working_output;
					output_size = temp_act->layer.input_cnt;
					working_output = new double[output_size];
					copyArray(temp_act->layer.getOutputs(), working_output, output_size);

					// If next is null, leave ptr at tail to traverse back to head for back propagation
					if (temp_act->next != nullptr) {
						temp_act = temp_act->next;
					}
				}
			}

			expected.getRow(expected_indexing[test - 1], expected_sample);
			display_error += error(expected_sample[0], working_output, output_size);

			double* working_error = new double[expected.cols];
			error_prime(expected_sample, working_output, working_error, expected.cols);

			// Back Propagate through network
			for (int curr_layer = layer_cnt; curr_layer >= 1; curr_layer--) {
				if (temp_conn != nullptr && temp_conn->layer.ID == curr_layer) {
					temp_conn->layer.backwardPropagation(working_error, learning_rate);

					delete[] working_error;
					working_error = new double[temp_conn->layer.input_cnt];
					copyArray(temp_conn->layer.getBPR(), working_error, temp_conn->layer.input_cnt);

					temp_conn = temp_conn->prev;
				}
				else if (temp_act != nullptr && temp_act->layer.ID == curr_layer) {
					temp_act->layer.backwardPropagation(working_error, learning_rate);

					delete[] working_error;
					working_error = new double[temp_act->layer.input_cnt];
					copyArray(temp_act->layer.getBPR(), working_error, temp_act->layer.input_cnt);

					temp_act = temp_act->prev;
				}
			}
			// Clear memory associated with the output/error to prepare for the next sample
			delete[] working_output;
			delete[] working_error;
		}

		display_error /= sample_cnt;

		//Calculate estimated time remaining
		auto epoch_end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed = epoch_end - epoch_start;
		std::chrono::duration<double> time_remaining = elapsed * (epochs - cnt);

		//Output average error to user every EPOCH_UPDATE_OCCURENCE iterations (for efficiency)
		if (cnt % EPOCH_UPDATE_OCCURENCE == 0) {
			std::cout << "\33[2K\r";
			std::cout << "Epoch " << cnt << "/" << epochs << ": error = " << display_error;
			std::cout << "   <Estimated Time Remaining: " << std::fixed << std::setprecision(2) << time_remaining.count() / 60 << " minutes>";
		}

		//Keep track of change in error, if small enough quit training the network
		if(init_error_delta){
			prev_error = display_error;
			init_error_delta = false;
		}
		else {
			double temp = abs(prev_error - display_error);
			if (error_delta > 0.0) {
				if (abs(temp - error_delta) < tolerance) {
					quit_level--;
				}
			}
			error_delta = temp;
		}
		if (quit_level == 0) {
			std::cout << "\nEarly Quit\n";
			break;
		}
	}
	
	// Release allocated memory
	delete[] expected_sample;
}

////////////////PRIVATE FUNCTIONS////////////////

ConnectedList* NeuralNetwork::clearConnectedList(ConnectedList *localPtr){
	if (localPtr != nullptr) {

		//Recurse until at the end of the list
		if (localPtr != nullptr) {
			clearConnectedList(localPtr->next);
		}

		//after recursive call, release memory for associated layer and localptr itself
		localPtr->layer.clear();
		delete localPtr;
	}
	return nullptr;
}

ActivationList* NeuralNetwork::clearActivationList(ActivationList *localPtr){
	if (localPtr != nullptr) {

		//Recurse until at the end of the list
		if (localPtr != nullptr) {
			clearActivationList(localPtr->next);
		}

		//after recursive call, release memory for associated layer and localptr itself
		localPtr->layer.clear();
		delete localPtr;
	}
	return nullptr;
}

int NeuralNetwork::load(std::string filename) {
	std::ifstream fh;
	fh.open(filename);
	if (!fh.is_open()) {
		std::cout << "[NeuralNetwork.load] Error Opening File, " << filename << " Not Found\n";
		return FAILURE;
	}
	std::string content = "";

	//Read through network header and grab relevant data
	while (fh >> content
		&& content != "----------------") {
		//Set ID
		if (content == "ID") {
			if (!(fh >> ID)) { return FAILURE; }
		}
		if (content == "Error") {
			if (!(fh >> errorName)) { return FAILURE; }
		}
		if (content == "Error_Prime") {
			if (!(fh >> errorPrimeName)) { return FAILURE; }
		}
	}

	//Loop through this networks layers
	while (fh >> content) {
		if (content == "Connected_Layer") {
			ConnectedLayer newConnLayer;
			//Load input and output amounts
			while (fh >> content && content != "Weights") {
				if (content == "Input") {
					if (!(fh >> newConnLayer.input_cnt)) { return FAILURE; }
				}
				if (content == "Output") {
					if (!(fh >> newConnLayer.output_cnt)) { return FAILURE; }
				}
			}
			newConnLayer.setupArrays(false);
			//Load weights
			int index = 0;
			while (index < newConnLayer.input_cnt * newConnLayer.output_cnt) {
				double temp;
				if (!(fh >> temp)) { return FAILURE; }
				newConnLayer.weights[index] = temp;
				index++;
			}
			//load Bias
			index = 0;
			fh >> content; //Skip past bias header
			while (index < newConnLayer.output_cnt) {
				if (!(fh >> newConnLayer.bias[index])) { return FAILURE; }
				index++;
			}
			addConnectedLayer(newConnLayer);
		}
		if (content == "Activation_Layer") {
			int input_output = 0;
			std::string activateName = "";
			std::string activatePrimeName = "";
			if (fh >> content && content == "Input/Output") {
				if (!(fh >> input_output)) { return FAILURE; }
			}
			if (fh >> content && content == "Activate") {
				if (!(fh >> activateName)) { return FAILURE; }
			}
			if (fh >> content && content == "Activate_Prime") {
				if (!(fh >> activatePrimeName)) { return FAILURE; }
			}
			ActivationLayer newActLayer(input_output, activateName, activatePrimeName);
			addActivationLayer(newActLayer);
		}
	}
	fh.close();
	return SUCCESS;
}

void NeuralNetwork::save(std::string filename) {
	std::ofstream fh; //file handle
	fh.open(filename);
	if (!fh.is_open()) {
		std::cout << "\n[NeuralNetwork.save] Error Opening '" << filename << "', Network Save Failed\n";
		return;
	}
	//Set up file headers
	fh << "Neural Network ID " << ID << std::endl;
	fh << "Layers " << layer_cnt << std::endl;
	fh << "Inputs " << initial_input_cnt << std::endl;
	fh << "Outputs " << final_output_cnt << std::endl;
	fh << "Error " << errorName << std::endl;
	fh << "Error_Prime " << errorPrimeName << std::endl;
	fh << "----------------\n";

	//save the member of each layer
	ConnectedList* temp_conn = conn_list;
	ActivationList* temp_act = act_list;
	for (int curr_layer = 1; curr_layer <= layer_cnt; curr_layer++) {
		if (temp_conn != nullptr && temp_conn->layer.ID == curr_layer) {
			temp_conn->layer.save(&fh);
			temp_conn = temp_conn->next;
		}
		else {
			if (temp_act != nullptr) {
				temp_act->layer.save(&fh);
				temp_act = temp_act->next;
			}
		}
		fh << std::endl;
	}
	fh.close();
}

#endif