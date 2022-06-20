// main.cpp : Started 5-20-21 by Kyle Larson and Kyle Savery for Proto6909
//TODO insert license here
#include <chrono>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "NN-utils.hpp"
#include "NeuralNetwork.hpp"
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/videoio.hpp>

//using namespace cv;

int main()
{

    // Get start time
    auto start = std::chrono::system_clock::now();

    //Mat src = imread("resources/square.jpg");
    //if (src.data == nullptr) {
    //    cout << "[main] Image could not be opened\n";
    //    return -1;
    //}
    //
    srand((unsigned)time(nullptr)); // Seed the rand function

    // Training Data
    const int cases = 8;
    const int input_size = 28;
    const int output_size = 4;
    double inputs_array[cases*input_size] = { 14, 6, 21, 3, 2, 14, 1, 13, 3, 27, 8, 12, 23, 14, 13, 20, 3, 8, 14, 13, 17, 13, 4, 18, 21, 18, 21, 20,
                                              27, 10, 27, 17, 26, 12, 4, 16, 9, 21, 21, 1, 5, 9, 19, 26, 1, 8, 18, 18, 3, 4, 9, 25, 17, 25, 9, 14,
                                              4, 28, 14, 25, 1, 20, 25, 7, 16, 21, 22, 23, 13, 21, 17, 8, 23, 14, 22, 9, 2, 3, 16, 7, 0, 10, 3, 8,
                                              12, 4, 27, 22, 11, 15, 15, 22, 21, 26, 23, 26, 26, 20, 20, 9, 25, 26, 25, 7, 12, 9, 19, 22, 1, 9, 13, 20,
                                              11, 4, 14, 8, 7, 15, 23, 20, 10, 22, 2, 14, 18, 23, 26, 28, 11, 1, 25, 3, 14, 18, 24, 28, 7, 15, 23, 21,
                                              22, 4, 21, 22, 1, 5, 20, 15, 25, 17, 24, 25, 7, 9, 10, 20, 21, 11, 23, 15, 5, 0, 22, 20, 8, 1, 5, 24,
                                              2, 14, 25, 16, 5, 28, 0, 15, 2, 24, 5, 8, 12, 8, 2, 3, 25, 22, 18, 22, 17, 16, 17, 27, 2, 21, 4, 12,
                                              13, 22, 7, 9, 13, 12, 8, 19, 12, 11, 22, 7, 15, 23, 28, 13, 24, 2, 8, 7, 25, 0, 12, 4, 5, 18, 1, 23};
    //double *inputs_array = new double[cases * input_size];
    //for (int i = 0; i < 16 * 5000; i++) {
    //    inputs_array[i] = rand() % 500 + 1;
    //}
    double expected_array[cases * output_size] = {  0, 0, 0, 0,
                                                    0, 0, 0, 1,
                                                    0, 0, 1, 0, 
                                                    0, 0, 1, 1, 
                                                    0, 1, 0, 0,
                                                    0, 1, 0, 1,
                                                    0, 1, 1, 0,
                                                    0, 1, 1, 1};
    Matrix input(cases, input_size, inputs_array);
    Matrix expected(cases, output_size, expected_array);
    double learning_rate = 0.1;
    int epochs = 1000;

    // Create the neural network
    //Load from file
    NeuralNetwork NN("NeuralNetworkData/NeuralNet_testing.properties");
    if(NN.isEmpty()){
        std::cout << "[main] Error loading Neural Network from file, program terminated\n";
        return -1;
    }

    //Create new
    //NeuralNetwork NN(1, "meanSquaredError", "meanSquaredErrorPrime");

    //// Create and add layers to the network
    //ConnectedLayer layer_one(input_size, 50);
    //ActivationLayer layer_two(50, "tan_hb", "tan_hb_prime");
    //ConnectedLayer layer_three(50, 25);
    //ActivationLayer layer_four(25, "tan_hb", "tan_hb_prime");
    //ConnectedLayer layer_five(25, output_size);
    //ActivationLayer layer_six(output_size, "tan_hb", "tan_hb_prime");

    //NN.addConnectedLayer(layer_one);
    //NN.addActivationLayer(layer_two);
    //NN.addConnectedLayer(layer_three);
    //NN.addActivationLayer(layer_four);
    //NN.addConnectedLayer(layer_five);
    //NN.addActivationLayer(layer_six);

    //// Training the network
    //NN.train(input, expected, epochs, learning_rate);

    // Use network to predict output based on input
    double* predicted = new double[output_size];
    double* test = new double[input_size];
    double* expected_result = new double[output_size];

    std::cout << "\nPrediction Results: \n";
    std::cout << "-------------------\n\n";
    for (int i = 0; i < cases; i++) {
        expected.getRow(i, expected_result);
        input.getRow(i, test);

        NN.predict(test, predicted, input_size);
        
        displayArray("Input: ", test, 1, input_size, false);
        displayArray("Expected Output: ", expected_result, 1, output_size, false);
        displayArray("Actual: ", predicted, 1, output_size, false);
        int index = compareActualToExpected(expected, predicted);
        expected.getRow(index, expected_result);
        int guess = convertBinToInt(expected_result, output_size);
        std::cout << "Guess: " << guess << std::endl;
        std::cout << std::endl;
    }

    //Shutdown the network by saving its data and releasing all dynamic allocation
    NN.shutdown("NeuralNetworkData/NeuralNet_testing.properties");

    // Clear allocated memory
    delete[] predicted;
    delete[] expected_result;
    delete[] test;

    // Get end time and display elapsed time
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\n\n=============================";
    std::cout << "\nElapsed Time (sec): " << elapsed.count() << std::endl;
    std::cout << "=============================\n";
    
    return 0; // Return Success
}