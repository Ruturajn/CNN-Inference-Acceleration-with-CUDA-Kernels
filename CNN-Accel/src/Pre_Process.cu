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
#include <CNNWeights_Layer3.h>
#include <CNN_Funcs.h>

using namespace std;
using namespace cv;

#define THREADSx 16
#define THREADSy 16


float pre_process(Mat &in_img){
    // Create events to time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float exec_time1, exec_time2, exec_time3, exec_time4;

    /***********************************Convolution Layer 1********************************/

    // Calculate the size of the image for allocating memory
    size_t img_size = (in_img.rows * in_img.cols * in_img.channels() * sizeof(float));

    // Defining device variables
    float *d_in_img, *d_layer1_wt, *d_layer1_bias, *d_layer1_conv_out, *h_layer1_conv_out;

    int num_rows_after_conv1 = num_rows_layer1 - (2 * (kernel_size_layer1/2)), num_cols_after_conv1 = num_cols_layer1 - (2 * (kernel_size_layer1/2));

    // Allocating memory
    cudaMalloc((void**)&d_in_img, img_size);
    cudaMalloc((void**)&d_layer1_conv_out, num_rows_after_conv1 * num_cols_after_conv1 * num_filters_layer1  * sizeof(float));
    cudaMalloc((void**)&d_layer1_wt, num_wt_elements_layer1*sizeof(float));
    cudaMalloc((void**)&d_layer1_bias, num_filters_layer1*sizeof(float));
    h_layer1_conv_out = (float *)malloc(num_rows_after_conv1 * num_cols_after_conv1 * num_filters_layer1  * sizeof(float));

    // Copying image and weights from host to device
    cudaMemcpy(d_in_img, in_img.ptr(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer1_wt, Layer1_Weights, num_wt_elements_layer1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer1_bias, Layer_1_Bias, num_filters_layer1*sizeof(float), cudaMemcpyHostToDevice);

    //Configuring threads and blocks
    int BLOCKSx = (num_rows_after_conv1 + THREADSx - 1) / THREADSx;
    int BLOCKSy = (num_cols_after_conv1 + THREADSy - 1) / THREADSy;

    dim3 threads(THREADSx, THREADSy);
    dim3 blocks(BLOCKSx, BLOCKSy);

    cout << in_img.rows  << "|" << in_img.cols << endl;
    // Launch the Layer 1 Convolution kernel
    cudaEventRecord(start);
    Conv_Kernel<<<blocks, threads>>>(d_in_img, d_layer1_wt,d_layer1_bias,
                                     num_rows_after_conv1, num_cols_after_conv1, 
                                     d_layer1_conv_out, kernel_size_layer1, 
                                     num_filters_layer1, num_channels_layer1);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time1, start, stop);

    //Print Layer 1 Convolution Exec time
    cout << "Time required to execute the kernel for Conv Layer 1 is : " << exec_time1 << endl;

    cudaMemcpy(h_layer1_conv_out, d_layer1_conv_out, num_rows_after_conv1 * num_cols_after_conv1 * num_filters_layer1  * sizeof(float), cudaMemcpyDeviceToHost);
    
    // for (int k=0;k<1;k++){
    //     for (int i=0;i<num_rows_after_conv1;i++){
    //         for (int j=0;j<1;j++){
    //             cout << h_layer1_conv_out[(i*num_cols_after_conv1 + j)*num_filters_layer1 + k] << ",";
    //         }
    //         cout << "" << endl;
    //     }
    // }
    // for (int i=0;i<num_rows_after_conv1;i++){
    //     cout << h_layer1_conv_out[(i*num_cols_after_conv1 + 1)*num_filters_layer1 + 0] << "," << i+1;
    //     cout << "" << endl;  
    // }

    

    /********************************Max-Pooling Layer 1********************************/

    // Calculate the size after max pooling
    // This is assuming input rows = input cols
    // Here '0' is the padding
    // Size = (((W-F) + 2*P)/S) + 1
    int size_after_max_pool = (((num_rows_after_conv1-pool_size) + 2*0)/pool_stride) + 1;
    
    //Configuring threads and blocks
    BLOCKSx = (size_after_max_pool + THREADSx - 1) / THREADSx;
    BLOCKSy = (size_after_max_pool + THREADSy - 1) / THREADSy;

    dim3 threads_1(THREADSx, THREADSy);
    dim3 blocks_1(BLOCKSx, BLOCKSy);

    cout << size_after_max_pool  << "|" << size_after_max_pool << endl;

    // Declare device variable for Max Pooling Output
    float *d_layer1_pool_out, *h_layer1_pool_out;
    cudaMalloc((void**)&d_layer1_pool_out, size_after_max_pool * size_after_max_pool * num_filters_layer1 * sizeof(float));

    h_layer1_pool_out = (float *)malloc(size_after_max_pool * size_after_max_pool * num_filters_layer1 * sizeof(float));
    // Launch the Max-Pooling Kernel
    cudaEventRecord(start);
    Max_Pool_Kernel<<<blocks_1, threads_1>>>(d_layer1_conv_out, pool_stride, 
                                            pool_size, d_layer1_pool_out,
                                            size_after_max_pool, num_rows_after_conv1,
                                            num_filters_layer1);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess){
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time2, start, stop);

    //Print Max Pool Layer 1 exec time
    std::cout << "Time required to execute the kernel for Max-Pooling is : " << exec_time2 << endl;

    cudaMemcpy(h_layer1_pool_out, d_layer1_pool_out, size_after_max_pool * size_after_max_pool * num_filters_layer1 * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i=0;i<size_after_max_pool;i++){
    //     cout << h_layer1_pool_out[(i*size_after_max_pool + 1)*num_filters_layer1 + 0] << "," << i+1;
    //     cout << "" << endl;  
    // }



    /***********************************Convolution Layer 2********************************/

    float *d_layer2_conv_out, *d_layer2_bias, *d_layer2_wt, *h_layer2_conv_out;
    int num_rows_after_max_pool = size_after_max_pool - (2 * (kernel_size_layer2 / 2 ));
    cudaMalloc((void**)&d_layer2_conv_out, num_rows_after_max_pool * num_rows_after_max_pool * num_filters_layer2 * sizeof(float));
    cudaMalloc((void**)&d_layer2_wt, num_wt_elements_layer2*sizeof(float));
    cudaMalloc((void**)&d_layer2_bias, num_filters_layer2*sizeof(float));
    h_layer2_conv_out = (float *)malloc(num_rows_after_max_pool * num_rows_after_max_pool * num_filters_layer2 * sizeof(float));

    cudaMemcpy(d_layer2_wt, Layer_2_Weights, num_wt_elements_layer2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer2_bias, Layer_2_Bias, num_filters_layer2*sizeof(float), cudaMemcpyHostToDevice);

    //Configuring threads and blocks
    BLOCKSx = (num_rows_after_max_pool + THREADSx - 1) / THREADSx;
    BLOCKSy = (num_rows_after_max_pool + THREADSy - 1) / THREADSy;

    dim3 threads_2(THREADSx, THREADSy);
    dim3 blocks_2(BLOCKSx, BLOCKSy);

    cout << num_rows_after_max_pool  << "|" << num_rows_after_max_pool << endl;

    // Launch the Layer 2 Convolution Kernel
    Conv_Kernel<<<blocks_2, threads_2>>>(d_layer1_pool_out, d_layer2_wt, d_layer2_bias, 
                                         num_rows_after_max_pool, num_rows_after_max_pool,
                                         d_layer2_conv_out, kernel_size_layer2,
                                         num_filters_layer2, num_channels_layer2);
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess){
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time3, start, stop);

    // Print the time for Conv Layer 2
    cout << "Time required to execute the kernel for Conv Layer 2 is : " << exec_time3 << endl;

    cudaMemcpy(h_layer2_conv_out, d_layer2_conv_out, num_rows_after_max_pool * num_rows_after_max_pool * num_filters_layer2 * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i=0;i<num_rows_after_max_pool;i++){
    //     cout << h_layer2_conv_out[(i*num_rows_after_max_pool + 1)*num_filters_layer2 + 0] << "," << i+1;
    //     cout << "" << endl;  
    // }

    /***********************************Dense FC Layer********************************/    

    // Launch the Dense layer kernel
    float *d_final_pred, *d_layer3_wt, *h_final_pred_arr, final_pred = 0.0f;
    cudaMalloc((void**)&d_layer3_wt, num_wt_elements_layer3 * sizeof(float));
    cudaMalloc((void**)&d_final_pred, num_wt_elements_layer3 * sizeof(float));
    h_final_pred_arr = (float *)malloc(num_wt_elements_layer3 * sizeof(float));
    cudaMemcpy(d_layer3_wt, Layer_3_Weights, num_wt_elements_layer3 * sizeof(float), cudaMemcpyHostToDevice);

    Dense_Layer_Kernel<<<blocks_2, threads_2>>>(d_layer2_conv_out, d_layer3_wt, d_final_pred,
                                                num_rows_after_max_pool, num_filters_layer2);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess){
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time4, start, stop);


    // After the computation, pass the answer through a sigmoid activation function.
    cudaMemcpy(h_final_pred_arr, d_final_pred, num_wt_elements_layer3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print time for fully connected dense layer
    cout << "Time required to execute the kernel for Dense Layer is : " << exec_time4 << endl;

    for (int i=0;i<num_wt_elements_layer3;i++){
        final_pred += h_final_pred_arr[i];
    }
    cout << "Final pred Before sigmoid : " << final_pred << endl;
    //cout << -1 * final_pred << endl;
    final_pred = 1 / (1 + exp(1.0 * final_pred));
    //cout << exp(-1.0 * final_pred) << endl;

    cout << "Predicted Val : " << final_pred << endl;
    cout << "Total Time on the GPU : " << exec_time1 + exec_time2 + exec_time3 + exec_time4 << endl;

    // Free the allocated memory
    cudaFree(d_in_img);
    cudaFree(d_layer1_wt);
    cudaFree(d_layer1_bias);
    cudaFree(d_layer1_conv_out);
    cudaFree(d_layer1_pool_out);
    cudaFree(d_layer2_wt);
    cudaFree(d_layer2_bias);
    cudaFree(d_layer2_conv_out);
    cudaFree(d_layer3_wt);
    free(h_final_pred_arr);
    free(h_layer1_conv_out);

    return final_pred;
}
