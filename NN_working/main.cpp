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

const char* image_folder = "resources\\training_data\\shapes\\*";

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
    //Disable openCV logging
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // Get start time
    auto start = std::chrono::system_clock::now();

    // Seed the rand function
    srand((unsigned)time(nullptr));

    //Setup the networks training data
    int* expected_indexing = nullptr;
    double* expected_array = nullptr;
    double* inputs_array = nullptr;
    ImageData* imageData = (ImageData*)calloc(1, sizeof(ImageData));
    if (imageData == nullptr) {
        std::cout << "[main] Failure creating 'imageData'\n";
        return -1;
    }

    //Use a folder full of images to initialize training data
    createTrainingData(image_folder, &inputs_array, &expected_array, &expected_indexing, imageData);

    if (inputs_array == nullptr 
        || expected_array == nullptr
        || expected_indexing == nullptr) {
        std::cout << "[main] Training Data failed to load from (" << image_folder << ")\n";
        return -1;
    }
    Matrix input(imageData->image_cnt, imageData->input_size, inputs_array);
    Matrix expected(imageData->num_outputs, imageData->output_size, expected_array);
    double learning_rate = 0.15;
    int epochs = 1000;

    //Create the neural network
    
    // Load from file
    /*NeuralNetwork NN("NeuralNetworkData/NeuralNet_testing.properties");
    if(NN.isEmpty()){
        std::cout << "[main] Error loading Neural Network from file, program terminated\n";
        return -1;
    }*/

    //Create new
    NeuralNetwork NN(1, "meanSquaredError", "meanSquaredErrorPrime");

    // Create and add layers to the network
    int first_layer_size = 16;
    int second_layer_size = 4;
    ConnectedLayer layer_one(imageData->input_size, first_layer_size);
    ActivationLayer layer_two(first_layer_size, "tan_hb", "tan_hb_prime");
    ConnectedLayer layer_three(first_layer_size, second_layer_size);
    ActivationLayer layer_four(second_layer_size, "tan_hb", "tan_hb_prime");
    ConnectedLayer layer_five(second_layer_size, imageData->output_size);
    ActivationLayer layer_six(imageData->output_size, "tan_hb", "tan_hb_prime");

    NN.addConnectedLayer(layer_one);
    NN.addActivationLayer(layer_two);
    NN.addConnectedLayer(layer_three);
    NN.addActivationLayer(layer_four);
    NN.addConnectedLayer(layer_five);
    NN.addActivationLayer(layer_six);

    // Training the network
    NN.train(input, expected, expected_indexing, epochs, learning_rate);

    //Use network to predict output based on input
    double* predicted = new double[imageData->output_size];
    double* test = new double[imageData->input_size];
    double* expected_result = new double[imageData->output_size];
    std::cout << "\nPrediction Results: \n";
    std::cout << "-------------------\n\n";
    std::string temp;
    for (int i = 0; i < imageData->image_cnt; i++) {
        //std::cin >> temp;
        expected.getRow(expected_indexing[i], expected_result);
        input.getRow(i, test);

        NN.predict(test, predicted, imageData->input_size);

        displayArray("Expected Output: ", expected_result, 1, imageData->output_size, false);
        displayArray("Actual: ", predicted, 1, imageData->output_size, false);
        std::cout << std::endl;
    }

    //Shutdown the network by saving its data and releasing all dynamic allocation
    NN.shutdown("NeuralNetworkData/NeuralNet_testing.properties");

    // Clear allocated memory
    delete[] predicted;
    delete[] expected_result;
    delete[] test;
    delete[] inputs_array;
    delete[] expected_array;
    delete[] expected_indexing;
    free(imageData);

    // Get end time and display elapsed time
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\n\n=============================";
    std::cout << "\nElapsed Time (sec): " << elapsed.count() << std::endl;
    std::cout << "=============================\n";
    
    return 0; // Return Success
}