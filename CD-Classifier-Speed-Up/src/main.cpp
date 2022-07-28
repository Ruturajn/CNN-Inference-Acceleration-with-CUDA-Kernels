#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

#include <CNNWeights_Layer1.h>
#include <CNNWeights_Layer2.h>
#include <CNNWeights_Layer3_128.h>
#include <CNNWeights_Layer4_1.h>

#define threshold 0.5

using namespace std;
using namespace cv;

float pre_process(Mat &in_img);

int main(int argc, char** argv){

        Mat h_img = imread(argv[1], 1);
        cout << h_img.cols << " || " << h_img.rows << endl;

        // Resize and normalize the input image
        Mat resized_in_image;
        h_img.convertTo(resized_in_image, CV_32FC3, 1.0/255.0, 0);
        resize(resized_in_image, resized_in_image, Size(64,64));

        float pred = pre_process(resized_in_image);
        if (pred >= threshold){
            putText(h_img, "Dog", Point(20,20), FONT_HERSHEY_COMPLEX, 1, Scalar(0,0,255),2);
            cout << "Dog Detected" << endl;
        }
        else {
            putText(h_img, "Cat", Point(20,20), FONT_HERSHEY_COMPLEX, 1, Scalar(0,0,255), 2);
            cout << "Cat Detected" << endl;
        }
        
        imshow("Detection",h_img);
        cv::waitKey(0);
    
    return 0;
}
