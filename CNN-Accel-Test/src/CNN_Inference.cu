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

__global__ void Conv_Kernel(float *d_in_img, float *d_layer_wt,
                            float *d_layer_bias, int num_rows,
                            int num_cols, float *d_layer_conv_out,
                            int kernel_size, int num_filters,
                            int num_channels){
        
    // Calculating row and column indices
    // These indices refer to the dimensions of the 
    // output image which will be reduced after convolution
    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);
    
    // Making sure we don't access stuff out of the image
    if (row < num_rows && col < num_cols){

        float prod = 0.0f;
        int img_index, img_index_out, row_val, col_val, filter_index, num_rows_img, num_cols_img, filter_jump;
        num_rows_img = num_rows + (2 * (kernel_size/2));
        num_cols_img = num_cols + (2 * (kernel_size/2));

        
        // Defining starting points so that the filter doesn't fall of the image
        // These variables define the strating points for the original image
        //int start_row = (row + (2 * (kernel_size/2))) - (kernel_size/2);
        //int start_col = (col + (2 * (kernel_size/2))) - (kernel_size/2);
        int start_row = row;
        int start_col = col;
        //int start_row = (row) - (kernel_size/2);
        //int start_col = (col) - (kernel_size/2);



        // For every filter in layer 1
        for (int i=0;i<num_filters;i++){
            
            // Initialize the product with the bias
            prod = d_layer_bias[i];

            // For every channel in the input image
            for (int j=0;j<num_channels;j++){

                // For every row in the filter
                for (int k=0;k<kernel_size;k++){
                    
                    row_val = start_row + k;
                    
                    // For every column in the filter
                    for (int l=0;l<kernel_size;l++){
                        col_val = start_col + l;

                        // If the filter overflows from the image area, multiply the out of bound
                        // coefficients with the nearest image pixel, hence changing the row_val
                        // and col_val values accordingly
                        // row_val = (row_val <= 0) ? 0 : ((row_val >= num_rows) ? (num_rows-1) : row_val);
                        // col_val = (col_val <= 0) ? 0 : ((col_val >= num_cols) ? (num_cols-1) : col_val);


                        if (row_val >= 0 && row_val < num_rows_img && col_val >= 0 && col_val < num_cols_img){
                            img_index = ((row_val*num_cols_img) + col_val)*num_channels + j;
                            filter_index = (k*kernel_size) + l + (j*kernel_size*kernel_size) + (i*num_channels*kernel_size*kernel_size);
                            prod += (d_in_img[img_index] * d_layer_wt[filter_index]);

                            if (threadIdx.x == 1 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0){
                                printf("Start_row , Start_col = %d,%d \n",start_row, start_col);
                                printf("row_val , col_val = %d, %d\n",row_val, col_val);
                                printf("image_val : %f | img_index : %d\n",d_in_img[img_index], img_index);
                                printf("filter_val : %f\n", d_layer_wt[filter_index]);
                            }
                        }
                    }
                }
            }
            img_index_out = ((row*num_cols) + col)*num_filters + i;
            // Do ReLU operation and write the output
            if (prod < 0)
                d_layer_conv_out[img_index_out] = 0;
            else
                d_layer_conv_out[img_index_out] = prod;
        }
    }
}

__global__ void Max_Pool_Kernel(float *d_layer_prev_conv_out, int stride, int max_pool_size,
                                float *d_layer_pool_out, int size_after_max_pool,
                                int num_rows_after_conv, int num_filters){
    // Calculating row and column indices
    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);
    
    // Making sure we don't access stuff out of the image
    if (row < size_after_max_pool && col < size_after_max_pool){

        int img_index, img_index_out, row_val, col_val;
        
        // Defining starting points so that the filter doesn't fall of the image
        int start_row = row*stride;
        int start_col = col*stride;

        // For every filter in layer 1
        for (int i=0;i<num_filters;i++){

            // Initialize max value with 0
            float max_val = 0.0f;

            // For every row in the filter
            for (int k=0;k<max_pool_size;k++){
                
                // For every column in the filter
                for (int l=0;l<max_pool_size;l++){
                    row_val = start_row + k;
                    col_val = start_col + l;
                    
                    // Check that we don't fall of the image
                    if (row_val >= 0 && row_val < num_rows_after_conv && col_val >= 0 && col_val < num_rows_after_conv){
                        img_index = ((row_val*num_rows_after_conv) + col_val)*num_filters + i;
                        if (max_val <= d_layer_prev_conv_out[img_index])
                            max_val = d_layer_prev_conv_out[img_index];
                    }
                }
            }
            // Write the output
            img_index_out = ((row*size_after_max_pool) + col)*num_filters + i;
            d_layer_pool_out[img_index_out] = max_val;
        }
    }
}

__global__ void Dense_Layer_Kernel(float *d_layer_conv_out, float *d_layer_wt,
                                   float *d_final_pred,
                                   int size_after_max_pool, int num_filters){

    // Calculating row and column indices
    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);

    // Making sure we don't access stuff out of the image
    if (row < size_after_max_pool && col < size_after_max_pool){

        int img_index, wt_index;
        //float prod = 0.0f;
        
        // For every filter in the previous conv layer
        for (int i=0;i<num_filters;i++){
            img_index = ((row * size_after_max_pool) + col)*num_filters + i;
            wt_index =  ((row * size_after_max_pool) + col) + (i*size_after_max_pool*size_after_max_pool);
            d_final_pred[wt_index] += d_layer_conv_out[img_index] * d_layer_wt[wt_index];
        }
    }
}
