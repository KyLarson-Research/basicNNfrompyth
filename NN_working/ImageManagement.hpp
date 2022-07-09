#ifndef IMAGE_MANAGEMENT_HPP
#define IMAGE_MANAGEMENT_HPP

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/matx.hpp>//opencv namespace = cv
#include <iostream>
#include "NN-utils.hpp"

//Function Prototypes
std::string getImageType(cv::Mat src);

int* convertImageToArray(std::string filename);

double* createTrainingData(std::string folder);

#endif