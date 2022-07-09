#include "ImageManagement.hpp"

//Load an image and convert it into a blurred grayscale single channel array
int* convertImageToArray(std::string filename) {

    //Open image
    cv::Mat src = cv::imread(filename);
    if (src.data == nullptr || src.type() != CV_8UC3) {
        return nullptr;
    }
    cv::GaussianBlur(src, src, cv::Size(3, 3), 0);
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

    int *array = new int[src.total()];
    int index = 0;
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            array[index++] = (int)(src.at<uchar>(row, col));
        }

    }
    return array;
}

double* createTrainingData(std::string folder){
    int* picOne = convertImageToArray("resources/20by20shapes/tinyblacksquare.jpg");
    int* picTwo = convertImageToArray("resources/20by20shapes/tinyblackhex.jpg");
    int* picThree = convertImageToArray("resources/20by20shapes/tinyblackcircle.jpg");
    int size = 3 * 400;
    double* inputs_array = new double[size];
    int nextValue;
    for (int i = 0; i < size; i++) {
        if (i < 400) {
            nextValue = picOne[i];
        }
        else if (i >= 400 && i < 800) {
            nextValue = picTwo[i - 400];
        }
        else {
            nextValue = picThree[i - 800];
        }
        inputs_array[i] = nextValue;
    }
    delete[] picOne;
    delete[] picTwo;
    delete[] picThree;

    return inputs_array;
}

//Get string represtation of image type
std::string getImageType(cv::Mat src)
{
    switch (src.type()) {
    case CV_8UC1: return "CV_8UC1";
                  break;
    case CV_8UC2: return "CV_8UC2";
                  break;
    case CV_8UC3: return "CV_8UC3";
                  break;
    case CV_8UC4: return "CV_8UC4";
                  break;
    case CV_16UC1: return "CV_16UC1";
                   break;
    case CV_16UC2: return "CV_16UC2";
                   break;
    case CV_16UC3: return "CV_16UC3";
                   break;
    case CV_16UC4: return "CV_16UC4";
                   break;
    case CV_16SC1: return "CV_16SC1";
                   break;
    case CV_16SC2: return "CV_16SC2";
                   break;
    case CV_16SC3: return "CV_16SC3";
                   break;
    case CV_16SC4: return "CV_16SC4";
                   break;
    case CV_32SC1: return "CV_32SC1";
                   break;
    case CV_32SC2: return "CV_32SC2";
                   break;
    case CV_32SC3: return "CV_32SC3";
                   break;
    case CV_32SC4: return "CV_32SC4";
                   break;
    case CV_32FC1: return "CV_32FC1";
                   break;
    case CV_32FC2: return "CV_32FC2";
                   break;
    case CV_32FC3: return "CV_32FC3";
                   break;
    case CV_32FC4: return "CV_32FC4";
                   break;
    case CV_64FC1: return "CV_64FC1";
                   break;
    case CV_64FC2: return "CV_64FC2";
                   break;
    case CV_64FC3: return "CV_64FC3";
                   break;
    case CV_64FC4: return "CV_64FC4";
                   break;
    default:
        return "CV_8UC1";

    }
}
