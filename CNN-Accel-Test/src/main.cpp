#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <CNNWeights_Layer1.h>
#include <CNNWeights_Layer2.h>
#include <CNNWeights_Layer3.h>

#define threshold 0.8

using namespace std;
using namespace cv;

float pre_process(Mat &in_img);

int main(int argc, char **argv){
    // Reading the input image
    Mat h_img = imread("../Images/dog_pic.jpeg", 1);
    //Mat h_img = imread(argv[1], 1);
    cout << h_img.cols << " || " << h_img.rows << endl;

    // Resize and normalize the input image
    Mat resized_in_image;
    //h_img.convertTo(resized_in_image, CV_32FC3, 1.0/255.0, 0);
    h_img.convertTo(resized_in_image, CV_32FC3);
    resize(resized_in_image, resized_in_image, Size(5,5));

    float *resized_img_data;
    resized_img_data = (float *)malloc(resized_in_image.rows * resized_in_image.cols * resized_in_image.channels() * sizeof(float));
    memcpy(resized_img_data, resized_in_image.ptr(), resized_in_image.rows * resized_in_image.cols * resized_in_image.channels() * sizeof(float));
    int num_rows = resized_in_image.rows, num_cols = resized_in_image.cols, num_channels = resized_in_image.channels();

    for (int k=0;k<num_channels;k++){
        for (int i=0;i<num_rows;i++){
            for (int j=0;j<num_cols;j++){
                cout << resized_img_data[(i*num_cols + j)*num_channels + k] << " ,";
            }
            cout << "" << endl;
        }
        cout << "" << endl;
    }

    float pred = pre_process(resized_in_image);
    // if (pred >= threshold){
    //     putText(h_img, "Dog", Point(10,20), FONT_HERSHEY_COMPLEX, 1, Scalar(255,255,255));
    //     cout << "Dog Detected" << endl;
    // }
    // else {
    //     putText(h_img, "Cat", Point(10,20), FONT_HERSHEY_COMPLEX, 1, Scalar(255,255,255));
    //     cout << "Cat Detected" << endl;
    // }
    
    //imshow("Detection",h_img);
    //cv::waitKey(0);

    return 0;
}
