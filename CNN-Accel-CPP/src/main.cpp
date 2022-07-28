#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <CNN_Funcs.h>
#include <CNNWeights_Layer1.h>
#include <CNNWeights_Layer2.h>
#include <CNNWeights_Layer3.h>

#define threshold 0.8

using namespace std;
//using namespace cv;

int main(int argc, char** argv){
    // Reading the input image
    cv::Mat h_img = cv::imread("/home/ruturajn/Downloads/test1/20.jpg", 1);
    //cv::Mat h_img = cv::imread(argv[1], 1);
    cout << h_img.cols << " || " << h_img.rows << endl;

    // Resize and normalize the input image
    cv::Mat resized_in_image;
    h_img.convertTo(resized_in_image, CV_32FC3, 1.0/255.0, 0);
    //h_img.convertTo(h_img, CV_32FC3);
    cv::resize(resized_in_image, resized_in_image, cv::Size(180,180));

    // float *resized_img_data;
    // resized_img_data = (float *)malloc(resized_in_image.rows * resized_in_image.cols * resized_in_image.channels() * sizeof(float));
    // cout << resized_in_image.rows * resized_in_image.cols * resized_in_image.channels() << endl;
    // memcpy(resized_img_data, resized_in_image.ptr(), resized_in_image.rows * resized_in_image.cols * resized_in_image.channels() * sizeof(float));
    
    // for (int k=0;k<1;k++){
    //     for (int i=0;i<resized_in_image.rows;i++){
    //         for (int j=0;j<1;j++){
    //             //cout << setw(10) << (int)resized_img_data[(i*resized_in_image.cols + j)*resized_in_image.channels() + k] << ",";
    //             cout << setw(10) << resized_img_data[(i*resized_in_image.cols + j)*resized_in_image.channels() + k] << ",";
    //             //cout << (i*resized_in_image.cols + j)*resized_in_image.channels() + k << endl;
    //             cout << i+1 << endl;
    //         }
    //     }
    // }

    float pred = pre_process(resized_in_image);

    if (pred >= threshold){
        cv::putText(h_img, "Dog", cv::Point(10,20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255,255,255));
        cout << "Dog Detected" << endl;
    }
    else {
        cv::putText(h_img, "Cat", cv::Point(10,20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255,255,255));
        cout << "Cat Detected" << endl;
    }
    
    //cv::imshow("Detection",h_img);
    //cv::waitKey(0);
    //free(resized_img_data);
    
    return 0;
}