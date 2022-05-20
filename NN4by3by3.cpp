// NN4by3by3.cpp : Authored 5-20-21 by Kyle Larson and Kyle Savery for Proto6909
//TODO insert license here
#include <iostream>
int dot(int* A, int* B, int n) {
    int dot = 0;
#
    for (int i = 0; i < n; i++) {
        dot += A[i] * B[i];
    }
    return dot;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));//may be a good spot to try some computations using exp to compare to python
}// e^x as in euler's function not some exponent but sometimes 2^x is substituted?

float sigmoid_derivative(float x) {
    return x * (1 - x);
}
int main()
{
    int test_inputs[16][4] = {
        {0, 0, 0, 0}, { 0, 0, 0, 1 }, { 0, 0, 1, 0 }, { 0, 0, 1, 1 }, { 1, 1, 0, 0 }, { 1, 1, 0, 1 }, { 1, 1, 1, 0 }, { 1, 1, 1, 1 },
        {0,1,0,0}, {0,1,0,1}, {0,1,1,0}, {0,1,1,1}, {1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1} };

    int test_outputs[8][3] = { {0, 0, 1}, {0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 1, 0 } };
    int epochs = 100000;
    float lr = 0.001;
    int inputLayerNeurons = 4;
    int hiddenLayerNeurons = 4;
    int outputLayerNeurons = 3;
    std::cout << "\ninput neurons" << inputLayerNeurons << "\nhidden neurons" << hiddenLayerNeurons << "\noutput neurons" << outputLayerNeurons;
    //Random weightsand bias initialization
    int   hidden_weights = np.random.uniform(size = (inputLayerNeurons, hiddenLayerNeurons))
        hidden_bias = np.random.uniform(size = (1, hiddenLayerNeurons))
        output_weights = np.random.uniform(size = (hiddenLayerNeurons, outputLayerNeurons))
        output_bias = np.random.uniform(size = (1, outputLayerNeurons))
    return 0;
}
