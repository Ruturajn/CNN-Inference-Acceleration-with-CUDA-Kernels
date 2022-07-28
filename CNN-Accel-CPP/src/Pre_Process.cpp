#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <chrono>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <CNNWeights_Layer1.h>
#include <CNNWeights_Layer2.h>
#include <CNNWeights_Layer3.h>
#include <CNN_Funcs.h>

using namespace std;
// using namespace cv;

#define THREADSx 16
#define THREADSy 16

float pre_process(cv::Mat &in_img)
{

    // Create events to time the function
    float exec_time1, exec_time2, exec_time3, exec_time4;

    /***********************************Convolution Layer 1********************************/

    // Calculate the size of the image for allocating memory
    size_t img_size = (in_img.rows * in_img.cols * in_img.channels() * sizeof(float));

    // Defining device variables
    float *d_in_img, *d_layer1_wt, *d_layer1_bias, *d_layer1_conv_out;

    int num_rows_after_conv1 = num_rows_layer1 - (2 * (kernel_size_layer1 / 2)), num_cols_after_conv1 = num_cols_layer1 - (2 * (kernel_size_layer1 / 2));

    // Allocating memory
    d_in_img = (float *)malloc(img_size);
    d_layer1_conv_out = (float *)malloc(num_rows_after_conv1 * num_cols_after_conv1 * num_filters_layer1 * sizeof(float));
    d_layer1_wt = (float *)malloc(num_wt_elements_layer1 * sizeof(float));
    d_layer1_bias = (float *)malloc(num_filters_layer1 * sizeof(float));

    // Copying image and weights from host to device
    memcpy(d_in_img, in_img.ptr(), img_size);
    memcpy(d_layer1_wt, Layer1_Weights, num_wt_elements_layer1 * sizeof(float));
    memcpy(d_layer1_bias, Layer_1_Bias, num_filters_layer1 * sizeof(float));

    cout << in_img.rows << "|" << in_img.cols << endl;

    // Launch the Layer 1 Convolution kernel
    auto start_cpu = std::chrono::high_resolution_clock::now();
    Conv_Kernel(d_in_img, d_layer1_wt, d_layer1_bias,
                num_rows_after_conv1, num_cols_after_conv1,
                d_layer1_conv_out, kernel_size_layer1,
                num_filters_layer1, num_channels_layer1);

    auto diff = std::chrono::high_resolution_clock::now() - start_cpu;
    auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
    exec_time1 = t1.count();

    // cv::Mat conv_layer1_img(num_rows_after_conv1, num_rows_after_conv1, CV_32FC1);
    // memcpy(conv_layer1_img.ptr(), d_layer1_conv_out, num_rows_after_conv1 * num_cols_after_conv1 * sizeof(float));
    // //imshow("Conv-1-Output", conv_layer1_img);
    // conv_layer1_img.convertTo(conv_layer1_img, CV_8UC1, 255.0, 0);
    // imwrite("Conv-1-Output.png", conv_layer1_img);
    //cv::waitKey(0);

    // Print Layer 1 Convolution Exec time
    cout << "Time required to execute the function for Conv Layer 1 is : " << exec_time1 << endl;

    // for (int i=0;i<num_rows_after_conv1;i++){
    //     cout << d_layer1_conv_out[(i*num_cols_after_conv1 + 1)*num_filters_layer1 + 0] << "," << i+1;
    //     cout << "" << endl;
    // }

    /********************************Max-Pooling Layer 1********************************/

    // Calculate the size after max pooling
    // This is assuming input rows = input cols
    // Here '0' is the padding
    // Size = (((W-F) + 2*P)/S) + 1
    int size_after_max_pool = (((num_rows_after_conv1 - pool_size) + 2 * 0) / pool_stride) + 1;

    cout << size_after_max_pool << "|" << size_after_max_pool << endl;

    // Declare device variable for Max Pooling Output
    float *d_layer1_pool_out;
    d_layer1_pool_out = (float *)malloc(size_after_max_pool * size_after_max_pool * num_filters_layer1 * sizeof(float));

    // Launch the Max-Pooling Kernel
    start_cpu = std::chrono::high_resolution_clock::now();
    Max_Pool_Kernel(d_layer1_conv_out, pool_stride,
                    pool_size, d_layer1_pool_out,
                    size_after_max_pool, num_rows_after_conv1,
                    num_filters_layer1);

    diff = std::chrono::high_resolution_clock::now() - start_cpu;
    t1 = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
    exec_time2 = t1.count();

    // Print Max Pool Layer 1 exec time
    std::cout << "Time required to execute the function for Max-Pooling is : " << exec_time2 << endl;
    // for (int i = 0; i < size_after_max_pool; i++)
    // {
    //     cout << d_layer1_pool_out[(i * size_after_max_pool + 1) * num_filters_layer1 + 0] << "," << i + 1;
    //     cout << "" << endl;
    // }

    /***********************************Convolution Layer 2********************************/

    float *d_layer2_conv_out, *d_layer2_bias, *d_layer2_wt;
    int num_rows_after_max_pool = size_after_max_pool - (2 * (kernel_size_layer2 / 2));
    d_layer2_conv_out = (float *)malloc(num_rows_after_max_pool * num_rows_after_max_pool * num_filters_layer2 * sizeof(float));
    d_layer2_wt = (float *)malloc(num_wt_elements_layer2 * sizeof(float));
    d_layer2_bias = (float *)malloc(num_filters_layer2 * sizeof(float));

    memcpy(d_layer2_wt, Layer_2_Weights, num_wt_elements_layer2 * sizeof(float));
    memcpy(d_layer2_bias, Layer_2_Bias, num_filters_layer2 * sizeof(float));

    cout << num_rows_after_max_pool << "|" << num_rows_after_max_pool << endl;

    // Launch the Layer 2 Convolution Kernel
    start_cpu = std::chrono::high_resolution_clock::now();
    Conv_Kernel(d_layer1_pool_out, d_layer2_wt, d_layer2_bias,
                num_rows_after_max_pool, num_rows_after_max_pool,
                d_layer2_conv_out, kernel_size_layer2,
                num_filters_layer2, num_channels_layer2);

    diff = std::chrono::high_resolution_clock::now() - start_cpu;
    t1 = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
    exec_time3 = t1.count();

    // Print the time for Conv Layer 2
    cout << "Time required to execute the function for Conv Layer 2 is : " << exec_time3 << endl;

    // for (int i = 0; i < num_rows_after_max_pool; i++)
    // {
    //     cout << d_layer2_conv_out[(i * num_rows_after_max_pool + 1) * num_filters_layer2 + 0] << "," << i + 1;
    //     cout << "" << endl;
    // }

    /***********************************Dense FC Layer********************************/

    // Launch the Dense layer kernel
    float *d_final_pred, *d_layer3_wt, final_pred = 0.0f;
    d_layer3_wt = (float *)malloc(num_wt_elements_layer3 * sizeof(float));
    d_final_pred = (float *)malloc(num_wt_elements_layer3 * sizeof(float));
    memcpy(d_layer3_wt, Layer_3_Weights, num_wt_elements_layer3 * sizeof(float));

    // Call the function
    start_cpu = std::chrono::high_resolution_clock::now();
    Dense_Layer_Kernel(d_layer2_conv_out, d_layer3_wt, d_final_pred,
                                                num_rows_after_max_pool, num_filters_layer2);

    diff = std::chrono::high_resolution_clock::now() - start_cpu;
    t1 = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
    exec_time4 = t1.count();

    // Print time for fully connected dense layer
    cout << "Time required to execute the function for Dense Layer is : " << exec_time4 << endl;

    for (int i=0;i<num_wt_elements_layer3;i++){
        final_pred += d_final_pred[i];
    }
    cout << "Final pred Before sigmoid : " << final_pred << endl;
    cout << -1 * final_pred << endl;
    final_pred = 1 / (1 + exp(-1.0 * final_pred));
    cout << exp(-1.0 * final_pred) << endl;

    cout << "Predicted Val : " << final_pred << endl;
    cout << "Total Time on the CPU : " << exec_time1 + exec_time2 + exec_time3 + exec_time4 << endl;

    // Free the allocated memory
    free(d_in_img);
    free(d_layer1_wt);
    free(d_layer1_bias);
    free(d_layer1_conv_out);
    free(d_layer1_pool_out);
    free(d_layer2_wt);
    free(d_layer2_bias);
    free(d_layer2_conv_out);
    free(d_layer3_wt);
    free(d_final_pred);

    return 0.0f;
}
