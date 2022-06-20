#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "Layer.hpp"
#include "NN-utils.hpp"
#include <iostream>
#include <fstream>
#include "Activations.hpp"

class ActivationLayer : public Layer{
	public:
		//int ID = 0;
		int input_cnt = 0;
		double (*activate)(double) = nullptr;
		double (*activate_prime)(double) = nullptr;
		std::string activateName = "";
		std::string activatePrimeName = "";
	private:
		double* input = nullptr;
		double* outputs = nullptr;
		double* outputs_prime = nullptr;
		double* back_propagation_result = nullptr;

	public:
		//Constructors
		ActivationLayer() {}

		ActivationLayer(int input_cnt, std::string activateName, std::string activatePrimeName);

		//~ActivationLayer();

		void backwardPropagation(double* out_err, double learning_rate);

		void clear();

		void display();

		//Calculate activated input
		void forwardPropagation(double* input);

		double* getOutputs();

		double* getBPR();

		void save(std::ofstream *fh);
		//void setID(int ID) { this->ID = ID; }
};

#endif
