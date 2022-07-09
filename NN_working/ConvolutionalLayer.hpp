#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include "ImageManagement.hpp" //opencv namespace = cv

//Function Prototypes
inline cv::Mat createFeatureMap(cv::Mat src) {
	cv::Mat feature_map;
	cv::GaussianBlur(src, feature_map, cv::Size(3, 3), 0);
	return feature_map;
}

#endif