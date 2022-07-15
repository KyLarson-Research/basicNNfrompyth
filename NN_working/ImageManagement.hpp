#ifndef IMAGE_MANAGEMENT_HPP
#define IMAGE_MANAGEMENT_HPP

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/matx.hpp>//openCV namespace = cv
#include <iostream>
#include <windows.h>
#include "NN-utils.hpp"

#include <fstream>

//Structs
struct ImageData {
    int image_cnt;
    int input_size;
    int output_size;
    int num_outputs;
};

//Constants
enum file_extensions{JPG, PNG, TDP, UNDEFINED};

const int MAX_STR_LEN = 250;

//Function Prototypes
std::string getImageType(cv::Mat src);

int* convertImageToArray(cv::Mat src);

void createTrainingData(const char* path, double** input, double** expected, 
                        int** expected_indexing, ImageData* imageData);

int getFileType(std::string filename);

int getExpectedIndex(std::string filename, Matrix expected);

#endif