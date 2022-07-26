#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "Layer.hpp"
#include "ConnectedLayer.hpp"
#include "ActivationLayer.hpp"
#include "Error.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

enum ReadResults{SUCCESS, FAILURE};

const int EPOCH_UPDATE_OCCURENCE = 1;

struct ConnectedList {
	ConnectedLayer layer;
	struct ConnectedList* next;
	struct ConnectedList* prev;
};

struct ActivationList {
	ActivationLayer layer;
	struct ActivationList* next;
	struct ActivationList* prev;
};

class NeuralNetwork{
	private:
		ConnectedList* conn_list = nullptr;
		ActivationList* act_list = nullptr;
		int ID = 0;
		int layer_cnt = 0;
		int final_output_cnt = 0;
		int initial_input_cnt = 0;
		double (*error)(double, double*, int) = nullptr;
		void (*error_prime)(double*, double*, double*, int) = nullptr;
		std::string errorName = "";
		std::string errorPrimeName = "";

	public:
		NeuralNetwork(std::string filename);

		NeuralNetwork(int ID, std::string errorName, std::string errorPrimeName);

		void addConnectedLayer(ConnectedLayer conn_layer);

		void addActivationLayer(ActivationLayer act_layer);

		void clear();

		void display(bool show_detail=false);

		bool isEmpty();

		void predict(double* input, double* dst, int input_size);

		void setError(double (*error)(double, double*, int), void (*error_prime)(double*, double*, double*, int));
		
		void shutdown(std::string filename);

		void train(Matrix inputs, Matrix expected, int *expected_indexing, int epochs, double learning_rate);

	private:
		ConnectedList* clearConnectedList(ConnectedList* localPtr);

		ActivationList* clearActivationList(ActivationList* localPtr);

		int load(std::string filename);

		void save(std::string filename);
};

#endif