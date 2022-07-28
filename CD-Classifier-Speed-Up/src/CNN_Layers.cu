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

float Conv_Layer(float *in_img, const float *layer_wt, const float *layer_bias,
                 int num_filters, int num_channels, int output_size,
                 float *layer_conv_out, int in_size, int num_wt_elements,
                 int kernel_size)
{

    // Create events to time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float exec_time = 0.0f;

    // Defining device variables
    float *d_in_img, *d_layer_wt, *d_layer_bias, *d_layer_conv_out;

    // Allocating memory
    cudaMalloc((void**)&d_in_img, in_size * in_size * num_channels * sizeof(float));
    cudaMalloc((void**)&d_layer_conv_out, output_size * output_size * num_filters  * sizeof(float));
    cudaMalloc((void**)&d_layer_wt, num_wt_elements*sizeof(float));
    cudaMalloc((void**)&d_layer_bias, num_filters*sizeof(float));

    // Copying image and weights from host to device
    cudaMemcpy(d_in_img, in_img, in_size * in_size * num_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_wt, layer_wt, num_wt_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_bias, layer_bias, num_filters*sizeof(float), cudaMemcpyHostToDevice);

    cout << output_size  << "|" << output_size << endl;

    //Configuring threads and blocks
    int BLOCKSx = (output_size + THREADSx - 1) / THREADSx;
    int BLOCKSy = (output_size + THREADSy - 1) / THREADSy;

    dim3 threads(THREADSx, THREADSy);
    dim3 blocks(BLOCKSx, BLOCKSy);

    // Launch the Layer 1 Convolution kernel
    cudaEventRecord(start);
    Conv_Kernel<<<blocks, threads>>>(d_in_img, d_layer_wt,d_layer_bias,
                                     output_size, output_size, 
                                     d_layer_conv_out, kernel_size, 
                                     num_filters, num_channels);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time, start, stop);

    //Print Layer 1 Convolution Exec time
    cout << "Time required to execute the kernel for Conv Layer is : " << exec_time << endl;

    cudaMemcpy(layer_conv_out, d_layer_conv_out, output_size * output_size * num_filters  * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the allocated memory
    cudaFree(d_in_img);
    cudaFree(d_layer_bias);
    cudaFree(d_layer_wt);
    cudaFree(d_layer_conv_out);

    // Return the exec. time
    return exec_time;
}

float Max_Pool_Layer(float *layer_prev_conv_out, int max_pool_size,
                    int max_pool_stride, float *max_pool_out,
                    int output_size, int num_filters,
                    int in_size){
    // Create events to time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float exec_time = 0.0f;
    
    //Configuring threads and blocks
    int BLOCKSx = (output_size + THREADSx - 1) / THREADSx;
    int BLOCKSy = (output_size + THREADSy - 1) / THREADSy;

    dim3 threads_1(THREADSx, THREADSy);
    dim3 blocks_1(BLOCKSx, BLOCKSy);

    cout << output_size  << "|" << output_size << endl;

    // Declare device variable for Max Pooling Output
    float *d_layer_pool_out, *d_layer_conv_out;
    cudaMalloc((void**)&d_layer_pool_out, output_size * output_size * num_filters * sizeof(float));
    cudaMalloc((void**)&d_layer_conv_out, in_size * in_size * num_filters * sizeof(float));
    cudaMemcpy(d_layer_conv_out, layer_prev_conv_out, in_size * in_size * num_filters * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the Max-Pooling Kernel
    cudaEventRecord(start);
    Max_Pool_Kernel<<<blocks_1, threads_1>>>(d_layer_conv_out, max_pool_stride, 
                                            max_pool_size, d_layer_pool_out,
                                            output_size, in_size,
                                            num_filters);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time, start, stop);

    //Print Max Pool Layer 1 exec time
    std::cout << "Time required to execute the kernel for Max-Pooling is : " << exec_time << endl;

    cudaMemcpy(max_pool_out, d_layer_pool_out, output_size * output_size * num_filters * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the allocated memory
    cudaFree(d_layer_pool_out);
    cudaFree(d_layer_conv_out);

    return exec_time;
}

float Dense_Layer(float *layer_prev_out,const float *layer_wt, const float *layer_bias,
                 float *layer_out, int in_size, int num_filters, int out_size,
                 int num_wt_elements, int num_flattened_elements){
    
    // Create events to time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float exec_time = 0.0f;

    float *d_final_pred, *d_layer_wt, *d_layer_prev_out, *d_layer_bias;
    cudaMalloc((void**)&d_layer_wt, num_wt_elements * sizeof(float));
    cudaMalloc((void**)&d_final_pred, out_size * sizeof(float));
    cudaMalloc((void**)&d_layer_prev_out, in_size * in_size * num_filters * sizeof(float));
    cudaMalloc((void**)&d_layer_bias, out_size * sizeof(float));

    cudaMemcpy(d_layer_prev_out, layer_prev_out, in_size * in_size * num_filters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_wt, layer_wt, num_wt_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_bias, layer_bias, out_size * sizeof(float), cudaMemcpyHostToDevice);

    //Configuring threads and blocks
    int BLOCKSx = (out_size + THREADSx - 1) / THREADSx;

    dim3 threads_2(THREADSx);
    dim3 blocks_2(BLOCKSx);

    cout << out_size  << "|" << out_size << endl;
    cudaEventRecord(start);
    Dense_Layer_Kernel<<<blocks_2, threads_2>>>(d_layer_prev_out, d_layer_wt, d_layer_bias,d_final_pred,
                                                in_size, out_size, num_flattened_elements);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time, start, stop);


    // After the computation, pass the answer through a sigmoid activation function.
    cudaMemcpy(layer_out, d_final_pred, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print time for fully connected dense layer
    cout << "Time required to execute the kernel for Dense Layer is : " << exec_time << endl;

    // Free the allocated memory
    cudaFree(d_layer_prev_out);
    cudaFree(d_layer_wt);
    cudaFree(d_final_pred);
    cudaFree(d_layer_bias);

    // Return exec. time
    return exec_time;
}

float Dense_Layer_Final(float *layer_prev_out, const float *layer_wt,
                        float *final_pred, int in_size, int num_wt_elements){
    // Create events to time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float exec_time = 0.0f;

    // Allocate memory and declare device variables
    float *d_layer_prev_out, *d_layer_wt, *d_final_pred;
    cudaMalloc((void**)&d_layer_prev_out, in_size * sizeof(float));
    cudaMalloc((void**)&d_layer_wt, num_wt_elements * sizeof(float));
    cudaMalloc((void**)&d_final_pred, in_size * sizeof(float));

    cudaMemcpy(d_layer_prev_out, layer_prev_out, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_wt, layer_wt, num_wt_elements * sizeof(float), cudaMemcpyHostToDevice);

    int BLOCKSx = (in_size + THREADSx - 1)/THREADSx;

    dim3 blocks(BLOCKSx);
    dim3 threads(THREADSx);

    cudaEventRecord(start);
    Dense_Layer_Final_Kernel<<<blocks, threads>>>(d_layer_prev_out, d_layer_wt, 
                                                  d_final_pred, in_size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time, start, stop);

    cudaMemcpy(final_pred, d_final_pred, in_size * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Time required to execute the kernel for Final Dense Layer is : " << exec_time << endl;

    // Free the allocated memory
    cudaFree(d_layer_prev_out);
    cudaFree(d_layer_wt);
    cudaFree(d_final_pred);

    //return exec. time
    return exec_time;
}


// ch 1 convolution (theoretical aspects of convolution, where is opportunity of parallelization,
// Lot of data processing , hence GPU)
// ch 2 GPU Architecture, Memory (details)
// ch 3 Efforts taken so far (Matrix , Blurring, Point Cloud, Graph Theory, based on this accelration techniques)
// ch 4 Acceleration of CNN and results
// ch 5 application of CNN in point cloud data (Point cloud is large data, CNN, big platform GPU requirement, lot of parallel)
