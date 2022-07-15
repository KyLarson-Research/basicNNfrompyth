#include "ImageManagement.hpp"

//Load an image and convert it into a blurred grayscale single channel array
int* convertImageToArray(cv::Mat src) {
    cv::resize(src, src, cv::Size(50, 50));
    cv::GaussianBlur(src, src, cv::Size(3, 3), 0);
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

    int *array = new int[src.rows * src.cols];
    int index = 0;
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            array[index] = (int)(src.at<uchar>(row, col));
            index++;
        }
    }
    return array;
}

void createTrainingData(const char* path, double** input, double** expected, 
                        int** expected_indexing, ImageData* imageData){
    
    std::vector<std::string> folder_contents;
    std::string properties_file;
    WIN32_FIND_DATAA filename;

    HANDLE handleFinder = FindFirstFileA(path, &filename);

    // Get all file names from desired path
    if (handleFinder != INVALID_HANDLE_VALUE){
        do{
            //Skip past any directories
            if (filename.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY) {
                continue;
            }
            //Save name for properties file
            if (getFileType(filename.cFileName) == TDP) {
                properties_file = filename.cFileName;
            }
            
            //Add name of each image to vector of image names
            else if (getFileType(filename.cFileName) == PNG
                || getFileType(filename.cFileName) == JPG) {
                folder_contents.push_back(filename.cFileName);
            }
            
        } while (FindNextFileA(handleFinder, &filename));

        FindClose(handleFinder);
    }
    else {
        FindClose(handleFinder);
        return;
    }

    std::string image_folder = "resources/training_data/shapes/";
    std::string temp = image_folder;

    //Load training data properties from file
    int width = 1, height = 1, num_images = 1, output_size = 1, total_output = 1;
    std::ifstream fh;
    temp += properties_file;
    fh.open(temp);
    if (!fh.is_open()) {
        std::cout << "[ImageManagement.createTrainingData] Training Data Properties failed to open (" << temp << ")\n";
        return;
    }

    // Loop through file and initialize training properties
    std::string content = "";
    while (fh >> content) {
        if (content == "WIDTH") {
            if (!(fh >> width)) { return; }
        }
        if (content == "HEIGHT") {
            if (!(fh >> height)) { return; }
        }
        if (content == "IMAGE_CNT") {
            if (!(fh >> num_images)) { return; }
        }
        if (content == "OUTPUT_SIZE") {
            if (!(fh >> output_size)) { return; }
        }
        if (content == "SIZE_OF_OUTPUT") {
            if (!(fh >> total_output)) { return; }
        }
        if (content == "EXPECTED_OUTPUT") {
            int index = 0;
            *expected = new double[total_output];
            while (index < total_output && fh >> content) {
                (*expected)[index] = stoi(content);
                index++;
            }

        }
    }
    fh.close();

    //Initialize imageData members
    imageData->image_cnt = num_images;
    imageData->input_size = width * height;
    imageData->output_size = output_size;
    imageData->num_outputs = total_output / output_size;

    //Initialize array for expected indexing
    *expected_indexing = new int[num_images];

    Matrix expected_matrix(total_output / output_size, output_size, *expected);

    //Fill the inputs array with data from each image
    *input = new double[num_images * width * height];

    temp = image_folder;
    cv::Mat src;
    for (int i = 0; i < num_images; i++) {

        //Attempt to open the image
        temp += folder_contents.at(i);
        src = cv::imread(temp);

        //If image is unsuccessfully opened or of the wrong type, fail and return a nullptr
        if (src.data == nullptr || src.type() != CV_8UC3) {
            std::cout << "[ImageManagement.createTrainingData] Image Failed to Open (" << temp << ")\n";
            delete[] *input;
            *input = nullptr;
            return;
        }
        //Check image dimensions against property files
        /*if (src.rows != height && src.cols != width) {
            std::cout << "[ImageManagement.createTrainingData] Image Dim Mismatch\n";
            delete[] *input;
            *input = nullptr;
            return;
        }*/

        (*expected_indexing)[i] = getExpectedIndex(folder_contents.at(i), expected_matrix);
        int* image_array = convertImageToArray(src);
        for (int k = 0; k < width*height; k++) {
           (*input)[k + (width*height*i)] = image_array[k];
        }
        std::cout << "\33[2K\r";
        std::cout << "Creating Training Data : (" << i+1 << " / " << num_images << ")";

        temp = image_folder;
        delete[] image_array;
    }
}

int getFileType(std::string filename){

    //Loop through string and extract file extension
    size_t length = filename.length();
    std::string extension = "";
    int index = 0, ext_index = 0;
    bool in_extension = false;
    while (index < length) {
        if (in_extension) {
            extension += filename[index];
        }
        if (filename[index] == '.') {
            in_extension = true;
        }
        index++;
    }
    
    //Determine file type
    if (extension.compare("tdp") == 0) {
        return TDP;
    }
    else if (extension.compare("png") == 0
            || extension.compare("PNG") == 0) {
        return PNG;
    }
    else if (extension.compare("jpg") == 0
             || extension.compare("JPG") == 0) {
        return JPG;
    }
    return UNDEFINED;
}

int getExpectedIndex(std::string filename, Matrix expected){
    double* expected_temp = new double[expected.cols];
    double* expected_file = new double[expected.cols];
    int index = 0;
    int copy_index = 0;
    bool underscore_found = false;
    while (index < filename.length() && copy_index < expected.cols && filename[index] != '_') {
        expected_file[copy_index] = (filename[index]) - '0';
        copy_index++;
        index++;
        if (filename[index] == '_') { underscore_found = true; }
    }
    if (!underscore_found) {
        delete[] expected_temp;
        delete[] expected_file;
        return -1;
    }

    int expected_index = 0;
    for (int row = 0; row < expected.rows; row++) {
        expected.getRow(row, expected_temp);
        bool same = true;
        for (int compare_ind = 0; compare_ind < expected.cols; compare_ind++) {
            if (expected_temp[compare_ind] != expected_file[compare_ind]) {
                same = false;
            }
        }
        if (same) {
            expected_index = row;
            break;
        }
    }

    delete[] expected_temp;
    delete[] expected_file;
    return expected_index;
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
