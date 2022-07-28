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

    // File Handling
    //fstream in_file;
    //in_file.open("Input_Image.txt",ios::out);

    //if (!in_file){
    //    cout << "File Creation Failed!!" << endl;
    //    return 0;
    //}
    //else{

        // Reading the input image
        //Mat h_img = imread("/home/ruturajn/Downloads/test1/20.jpg", 1);
        Mat h_img = imread(argv[1], 1);
        cout << h_img.cols << " || " << h_img.rows << endl;

        // Resize and normalize the input image
        Mat resized_in_image;
        h_img.convertTo(resized_in_image, CV_32FC3, 1.0/255.0, 0);
        //h_img.convertTo(resized_in_image, CV_32FC3);
        //resize(resized_in_image, resized_in_image, Size(64,64), INTER_MAX);
        resize(resized_in_image, resized_in_image, Size(64,64));

        //float *resized_img_data;
        //resized_img_data = (float *)malloc(resized_in_image.rows * resized_in_image.cols * resized_in_image.channels() * sizeof(float));
        //memcpy(resized_img_data, resized_in_image.ptr(), resized_in_image.rows * resized_in_image.cols * resized_in_image.channels() * sizeof(float));

        // for (int i=0;i<resized_in_image.rows;i++){
        //     cout << resized_img_data[(i*resized_in_image.cols + 0)*resized_in_image.channels() + 0] << "," << i+1;
        //     cout << "" << endl;
        // }
        // for (int k=0;k<resized_in_image.channels();k++){
        //     for (int i=0;i<resized_in_image.rows;i++){
        //         for (int j=0;j<resized_in_image.cols;j++){
        //             in_file << resized_img_data[(i*resized_in_image.cols + j)*resized_in_image.channels() + k] << ",";
        //         }
        //         in_file << "" << endl;
        //     }
        //     in_file << "" << endl;
        // }

        // in_file.close();

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
        //free(resized_img_data);
    //}
    
    return 0;
}
