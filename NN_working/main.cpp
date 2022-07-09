// main.cpp : Started 5-20-21 by Kyle Larson and Kyle Savery for Proto6909
//TODO insert license here
#include <chrono>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "NeuralNetwork.hpp"
#include "ImageManagement.hpp"
#include "ConvolutionalLayer.hpp"

void decToBinary(int n, double *dst, int bits)
{
    // counter for binary array
    int i = bits-1;
    while (n > 0) {

        // storing remainder in binary array
        dst[i] = n % 2;
        n = n / 2;
        i--;
    }
    while (i >= 0) {
        dst[i] = 0;
        i--;
    }
}

int main()
{

    // Get start time
    auto start = std::chrono::system_clock::now();

    srand((unsigned)time(nullptr)); // Seed the rand function

    // Training Data
    const int cases = 3;
    const int input_size = 20*20;
    const int output_size = 2;
    const int num_outputs = 3;

    int expected_indexing[num_outputs * output_size] = {0, 1, 2}; //Indicates which output the inputs correspond to
    
    double *inputs_array = createTrainingData("resources/20by20shapes");

    double expected_array[] = {0, 0, 0, 1, 1, 0};

    Matrix input(cases, input_size, inputs_array);
    Matrix expected(num_outputs, output_size, expected_array);
    double learning_rate = 0.1;
    int epochs = 3000;

    // Create the neural network
    
    // Load from file
  /*  NeuralNetwork NN("NeuralNetworkData/NeuralNet_testing.properties");
    if(NN.isEmpty()){
        std::cout << "[main] Error loading Neural Network from file, program terminated\n";
        return -1;
    }*/

    //Create new
    NeuralNetwork NN(1, "meanSquaredError", "meanSquaredErrorPrime");

    // Create and add layers to the network
    int first_layer_size = 50;
    int second_layer_size = 25;
    ConnectedLayer layer_one(input_size, first_layer_size);
    ActivationLayer layer_two(first_layer_size, "tan_hb", "tan_hb_prime");
    ConnectedLayer layer_three(first_layer_size, second_layer_size);
    ActivationLayer layer_four(second_layer_size, "tan_hb", "tan_hb_prime");
    ConnectedLayer layer_five(second_layer_size, output_size);
    ActivationLayer layer_six(output_size, "tan_hb", "tan_hb_prime");

    NN.addConnectedLayer(layer_one);
    NN.addActivationLayer(layer_two);
    NN.addConnectedLayer(layer_three);
    NN.addActivationLayer(layer_four);
    NN.addConnectedLayer(layer_five);
    NN.addActivationLayer(layer_six);

    // Training the network
    NN.train(input, expected, expected_indexing, epochs, learning_rate);

    //Use network to predict output based on input
    double* predicted = new double[output_size];
    double* test = new double[input_size];
    double* expected_result = new double[output_size];
    std::cout << "\nPrediction Results: \n";
    std::cout << "-------------------\n\n";
    std::string temp;
    for (int i = 0; i < cases; i++) {
        std::cin >> temp;
        expected.getRow(expected_indexing[i], expected_result);
        input.getRow(i, test);

        NN.predict(test, predicted, input_size);

        displayArray("Expected Output: ", expected_result, 1, output_size, false);
        displayArray("Actual: ", predicted, 1, output_size, false);
        std::cout << std::endl;
    }

    //Shutdown the network by saving its data and releasing all dynamic allocation
    NN.shutdown("NeuralNetworkData/NeuralNet_testing.properties");

    // Clear allocated memory
    delete[] predicted;
    delete[] expected_result;
    delete[] test;
    delete[] inputs_array;

    // Get end time and display elapsed time
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\n\n=============================";
    std::cout << "\nElapsed Time (sec): " << elapsed.count() << std::endl;
    std::cout << "=============================\n";
    
    return 0; // Return Success
}