#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/time.h>

#include <CNNWeights_Layer1.h>
#include <CNNWeights_Layer2.h>
#include <CNNWeights_Layer3_128.h>
#include <CNNWeights_Layer4_1.h>
#include <CNN_Funcs.h>


// Dense Layer Kernel (In CNN_Inference.cu)
void Dense_Layer_Kernel(float *d_layer_conv_out, const float *d_layer_wt, const float *d_layer_bias,
                        float *d_pred, int in_size,
                        int num_dense_elements, int num_flattened_elements);

// Final Dense Layer Kernel (In CNN_Inference.cu)
void Dense_Layer_Final_Kernel(float *d_layer_prev_out, const float *d_layer_wt,
                              float *d_final_pred, int in_size);

using namespace std;
using namespace cv;


float Conv_Layer(float *in_img, const float *layer_wt, const float *layer_bias,
                 int num_filters, int num_channels, int output_size,
                 float *layer_conv_out, int in_size, int num_wt_elements,
                 int kernel_size)
{

    // Create events to time the kernel
    struct timeval t1, t2;

    cout << output_size  << "|" << output_size << endl;

    // Launch the Layer 1 Convolution Function 
    gettimeofday(&t1,0);
    Conv_Kernel(in_img, layer_wt,layer_bias,
                output_size, output_size, 
                layer_conv_out, kernel_size, 
                num_filters, num_channels);
    gettimeofday(&t2,0);
    float exec_time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0; // Time taken by kernel in seconds

    //Print Layer 1 Convolution Exec time
    cout << "Time required to execute the function for Conv Layer is : " << exec_time << endl;


    // Return the exec. time
    return exec_time;
}

float Max_Pool_Layer(float *layer_prev_conv_out, int max_pool_size,
                    int max_pool_stride, float *max_pool_out,
                    int output_size, int num_filters,
                    int in_size){
    // Create events to time the kernel

    struct timeval t1, t2;
    gettimeofday(&t1,0);
    Max_Pool_Kernel(layer_prev_conv_out, max_pool_stride, 
                    max_pool_size, max_pool_out,
                    output_size, in_size,
                    num_filters);
    gettimeofday(&t2,0);
    float exec_time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0; // Time taken by kernel in seconds

    //Print Max Pool Layer 1 exec time
    std::cout << "Time required to execute the kernel for Max-Pooling is : " << exec_time << endl;

    return exec_time;
}

float Dense_Layer(float *layer_prev_out,const float *layer_wt, const float *layer_bias,
                 float *layer_out, int in_size, int num_filters, int out_size,
                 int num_wt_elements, int num_flattened_elements){
    
    // Create events to time the kernel

    struct timeval t1, t2;
    gettimeofday(&t1,0);
    Dense_Layer_Kernel(layer_prev_out, layer_wt, layer_bias, layer_out,
                       in_size, out_size, num_flattened_elements);

    gettimeofday(&t2,0);
    // Print time for fully connected dense layer

    float exec_time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0; // Time taken by kernel in seconds

    cout << "Time required to execute the kernel for Dense Layer is : " << exec_time << endl;

    // Return exec. time
    return exec_time;
}

float Dense_Layer_Final(float *layer_prev_out, const float *layer_wt,
                        float *final_pred, int in_size, int num_wt_elements){ 
    
    struct timeval t1, t2;
    gettimeofday(&t1,0);
    Dense_Layer_Final_Kernel(layer_prev_out, layer_wt, 
                             final_pred, in_size);

    gettimeofday(&t2,0);
    float exec_time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0; // Time taken by kernel in seconds

    cout << "Time required to execute the kernel for Final Dense Layer is : " << exec_time << endl; 

    //return exec. time
    return exec_time;
}
