// NN4by3by3.cpp : Authored 5-20-21 by Kyle Larson and Kyle Savery for Proto6909
//TODO insert license here
#include <iostream>
#include <time.h> //time
#include <stdlib.h> //rand

float dot(float* A, float* B, int n) {
    float dot = 0;
#
    for (int i = 0; i < n; i++) {
        dot += A[i] * B[i];
    }
    return dot;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));//may be a good spot to try some computations using exp to compare to python
}// e^x as in euler's function not some exponent but sometimes 2^x is substituted?

float working_error() {

}

float sigmoid_derivative(float x) {
    float temp = sigmoid(x);
    return temp * (1 - temp);
}
float sq(float x) { return x * x; }
float mean_sq_err(float *e, float *o, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) { sum += sq( o[i] - e[i] ); }// sum squared error
    return sum / n;
}


int main()
{
    

    int test_inputs[4][2] = { {-1,-1}, {-1,1}, {1,-1}, {1,1} };

        /*{ {0, 0, 0, 0}, { 0, 0, 0, 1 }, { 0, 0, 1, 0 }, 
        { 0, 0, 1, 1 }, { 1, 1, 0, 0 }, { 1, 1, 0, 1 }, 
        { 1, 1, 1, 0 }, { 1, 1, 1, 1 }, { 0, 1, 0, 0 }, 
        { 0, 1, 0, 1 }, { 0, 1, 1, 0 }, { 0, 1, 1, 1 }, 
        { 1, 0, 0, 0 }, { 1, 0, 0, 1 }, { 1, 0, 1, 0 }, { 1, 0, 1, 1} };*/

    int test_outputs[4][1] = { 1, -1, -1, 1 };
        /*{ {0, 0, 1}, {0, 0, 0}, {1, 0, 1}, 
        {0, 1, 1}, {1, 1, 0}, {1, 0, 0},
        {1, 1, 1}, {0, 1, 0 } };*/
    // initial quanities
    int epochs = 100000;
    float lr = 0.001;
    int inputLayerNeurons = 2; 
    int hiddenLayerNeurons = 2;
    int outputLayerNeurons = 1;
    
    //int
    float i_1 = test_inputs[0][0];
    float i_2 = test_inputs[0][1];
    float I[2] = { i_1, i_2 };
    //Random weights and bias initialization
    //initialize random
    srand(time(NULL));
    float w_1 = ((float)rand() / RAND_MAX) * 2 - 1; //Generate random numbers between -1 and 1
    float w_2 = ((float)rand() / RAND_MAX) * 2 - 1;
    float w_3 = ((float)rand() / RAND_MAX) * 2 - 1;
    float w_4 = ((float)rand() / RAND_MAX) * 2 - 1;
    float w_5 = ((float)rand() / RAND_MAX) * 2 - 1;
    float w_6 = ((float)rand() / RAND_MAX) * 2 - 1;

    float u[2] = { w_1, w_2 };
    float v[2] = { w_3, w_4 };
    float w[2] = { w_5, w_6 };

    // initial forward propigation

    float h_1 = sigmoid(dot(u, I, 2));
    float h_2 = sigmoid(dot(v, I, 2));
    float h[2] = { h_1, h_2 };
    float out = sigmoid(dot(h, w, 2));

    // back propigate 
   // float working_error = 

    std::cout 
        << "\ninput: " << i_1 << " " << i_2
        << "\nn input neurons: " << inputLayerNeurons
        << "\nhidden input weight: " << w_1 << "  " << w_2 << "  " << w_3 << "  " << w_4
        << "\nn hidden neurons: " << hiddenLayerNeurons
        << "\nhidden neuron potentials: " << h_1 << " " << h_2
        << "\nhidden output weight: " << w_5 << "  " << w_6 
        << "\nn output neurons: " << outputLayerNeurons
        << "\noutput: " << out
        << "\n\n";

    

    //int   hidden_weights = 
        
    return 0;
}
//________________SECTION FOR PORTING FROM PYTHON TO THE ABOVE
/*import numpy as np 
#from math import atan
#np.random.seed(0)

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#Input datasets
inputs = np.array(test_inputs)#Day
expected_output = np.array(test_outputs)#e_e

epochs = 100000
lr = 0.001
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 4,4,3 #originally 2,2,1

#Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))

print("Initial hidden weights: ",end='')
print(*hidden_weights)
print("Initial hidden biases: ",end='')
print(*hidden_bias)
print("Initial output weights: ",end='')
print(*output_weights)
print("Initial output biases: ",end='')
print(*output_bias)


#Training algorithm
for _ in range(epochs):
    #Forward Propagation
    hidden_layer_activation = np.dot(inputs,hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output,output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    #Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    #Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

print("Final hidden weights: ",end='')
print(*hidden_weights)
print("Final hidden bias: ",end='')
print(*hidden_bias)
print("Final output weights: ",end='')
print(*output_weights)
print("Final output bias: ",end='')
print(*output_bias)

print("\nOutput from neural network after "+str(epochs)+" epochs: ",end='')
print(*predicted_output)
print(test_outputs)
*/