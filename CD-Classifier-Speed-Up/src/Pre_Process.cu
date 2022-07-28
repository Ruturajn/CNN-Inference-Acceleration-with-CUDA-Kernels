#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <CNNWeights_Layer1.h>
#include <CNNWeights_Layer2.h>
#include <CNNWeights_Layer3_128.h>
#include <CNNWeights_Layer4_1.h>
#include <CNN_Funcs.h>

using namespace std;
using namespace cv;

#define THREADSx 16
#define THREADSy 16

float pre_process(Mat &in_img)
{
    // Copy the contents of the input image to an array
    float *arr_in_img, *conv_layer1_out, total_time, exec_time1, exec_time2, exec_time3, exec_time4, exec_time5, exec_time6;
    size_t img_size = (in_img.rows * in_img.cols * in_img.channels() * sizeof(float));
    
    /***********************************Convolution Layer 1********************************/
    arr_in_img = (float *)malloc(img_size);
    memcpy(arr_in_img, in_img.ptr(), img_size);
    int conv_layer_1_output_size = size_layer1 - kernel_size_layer1 + 1;
    conv_layer1_out = (float *)malloc(conv_layer_1_output_size * conv_layer_1_output_size * num_filters_layer1 * sizeof(float));
    
    exec_time1 = Conv_Layer(arr_in_img, Layer1_Weights, Layer_1_Bias, num_filters_layer1,
                            num_channels_layer1, conv_layer_1_output_size,
                            conv_layer1_out, size_layer1, num_wt_elements_layer1, kernel_size_layer1);


    /********************Extracting Per Channel Outputs and storing them in a Image*********************/
    // cv::Mat conv_layer1_img(conv_layer_1_output_size, conv_layer_1_output_size, CV_32FC1);
    // memcpy(conv_layer1_img.ptr(), conv_layer1_out, conv_layer_1_output_size * conv_layer_1_output_size * sizeof(float));
    // //imshow("Conv-1-Output", conv_layer1_img);
    // conv_layer1_img.convertTo(conv_layer1_img, CV_8UC3, 255.0, 0);
    // imwrite("Conv-1-Output.png", conv_layer1_img);

    // for (int i=0;i<conv_layer_1_output_size;i++){
    //    cout << conv_layer1_out[(i*conv_layer_1_output_size + 3)*num_filters_layer1 + 1] << "," << i+1 << endl;
    // }


    /********************************Max-Pooling Layer 1********************************/

    // Calculate the size after max pooling
    // This is assuming input rows = input cols
    // Here '0' is the padding
    // Size = (((W-F) + 2*P)/S) + 1
    int max_pool_out_size_layer1 = (((conv_layer_1_output_size-pool_size) + 2*0)/pool_stride) + 1;
    float *max_pool_layer1_out;
    max_pool_layer1_out = (float *)malloc(max_pool_out_size_layer1 * max_pool_out_size_layer1 * num_filters_layer1 * sizeof(float));
    
    exec_time2 = Max_Pool_Layer(conv_layer1_out, pool_size, pool_stride,
                                max_pool_layer1_out, max_pool_out_size_layer1,
                                num_filters_layer1, conv_layer_1_output_size);

    // for (int i=0;i<max_pool_out_size_layer1;i++){
    //     cout << max_pool_layer1_out[(i*max_pool_out_size_layer1 + 0)*num_filters_layer1 + 1] << "," << i+1 << endl;
    // }

    /***********************************Convolution Layer 2********************************/

    // Size after 2nd Convolution Layer
    int conv_layer_2_output_size = max_pool_out_size_layer1 - kernel_size_layer2 + 1;

    float *conv_layer2_out;
    conv_layer2_out = (float *)malloc(conv_layer_2_output_size * conv_layer_2_output_size * num_filters_layer2 * sizeof(float));

    exec_time3 = Conv_Layer(max_pool_layer1_out, Layer_2_Weights, Layer_2_Bias, num_filters_layer2,
                            num_channels_layer2, conv_layer_2_output_size,
                            conv_layer2_out, max_pool_out_size_layer1, num_wt_elements_layer2, kernel_size_layer2);
    
    // for (int i=0;i<conv_layer_2_output_size;i++){
    //     cout << conv_layer2_out[(i*conv_layer_2_output_size + 0)*num_filters_layer2 + 1] << "," << i+1 << endl;
    // }
    

    /********************************Max-Pooling Layer 2********************************/
    
    int max_pool_out_size_layer2 = (((conv_layer_2_output_size-pool_size) + 2*0)/pool_stride) + 1;
    float *max_pool_layer2_out;
    max_pool_layer2_out = (float *)malloc(max_pool_out_size_layer2 * max_pool_out_size_layer2 * num_filters_layer2 * sizeof(float));

    exec_time4 = Max_Pool_Layer(conv_layer2_out, pool_size, pool_stride, 
                                max_pool_layer2_out, max_pool_out_size_layer2,
                                num_filters_layer2, conv_layer_2_output_size);
    
    // for (int i=0;i<max_pool_out_size_layer2;i++){
    //     //for (int j=0;j<max_pool_out_size_layer2;j++){
    //         cout << max_pool_layer2_out[(i*max_pool_out_size_layer2 + 1)*num_filters_layer2 + 0] << ",";
    //     //}
    //     cout << "" << endl;
    // }

    /***********************************Dense FC Layer 1********************************/

    // Flattening the output from the previous Layer
    float *flattened_arr;
    flattened_arr = (float *)malloc(max_pool_out_size_layer2 * max_pool_out_size_layer2 * num_channels_layer2 * sizeof(float));
    int index_flattened, index_before_flatten;
    
    for (int k=0;k<num_channels_layer2;k++){
        for (int i=0;i<max_pool_out_size_layer2;i++){
            for (int j=0;j<max_pool_out_size_layer2;j++){
                index_flattened = (i*max_pool_out_size_layer2) + j + (k*max_pool_out_size_layer2*max_pool_out_size_layer2);
                index_before_flatten = ((i*max_pool_out_size_layer2) + j)*num_channels_layer2 + k;
                flattened_arr[index_flattened] = max_pool_layer2_out[index_before_flatten];
            }
        }
    }
    
    float *dense_layer1_out;
    dense_layer1_out = (float *)malloc(num_dense_layer1 * sizeof(float));

    exec_time5 = Dense_Layer(flattened_arr, Layer_3_Weights, Layer_3_Bias,dense_layer1_out,
                             max_pool_out_size_layer2, num_filters_layer2, num_dense_layer1, 
                             num_wt_elements_layer3, num_flattened);

    //for (int i=0;i<num_dense_layer1;i++){
    //    cout << dense_layer1_out[i] << endl;
    //}


    /***********************************Dense FC Layer 2********************************/

    float *final_pred, prediction = Layer_4_Bias;
    final_pred = (float *)malloc(num_dense_layer2 * sizeof(float));

    exec_time6 = Dense_Layer_Final(dense_layer1_out, Layer_4_Weights, final_pred,
                                    num_dense_layer2, num_dense_layer2);
    

    for (int i=0;i<num_dense_layer2;i++){
        prediction += final_pred[i];
    }

    total_time = exec_time1 + exec_time2 + exec_time3 + exec_time4 + exec_time5 + exec_time6;
    cout << "The Total Time on GPU : " << total_time << endl;

    //prediction += Layer_4_Bias;
    cout << "Final Pred before sigmoid : " << prediction << endl;
    prediction = (1 / (1 + exp(-1.0 * prediction)));
    cout << "Final Pred after sigmoid : " << prediction << endl;
    

    // Free the allocated memory
    free(arr_in_img);
    free(conv_layer1_out);
    free(max_pool_layer1_out);
    free(conv_layer2_out);
    free(max_pool_layer2_out);
    free(dense_layer1_out);
    free(final_pred);
    
    return prediction;
}

