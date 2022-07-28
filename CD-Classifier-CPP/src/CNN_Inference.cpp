#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <CNNWeights_Layer1.h>
#include <CNNWeights_Layer2.h>
#include <CNNWeights_Layer3_128.h>
#include <CNNWeights_Layer4_1.h>
#include <CNN_Funcs.h>

using namespace std;
using namespace cv;


void Conv_Kernel(float *d_in_img, const float *d_layer_wt,
                 const float *d_layer_bias, int num_rows,
                 int num_cols, float *d_layer_conv_out,
                 int kernel_size, int num_filters,
                 int num_channels){
        
    for (int row=0;row<num_rows;row++){    
        for (int col=0;col<num_cols;col++){
        float prod = 0.0f;
        int img_index, img_index_out, row_val, col_val, filter_index, num_rows_img, num_cols_img, filter_jump;
        num_rows_img = num_rows + (2 * (kernel_size/2));
        num_cols_img = num_cols + (2 * (kernel_size/2));

        
        // Defining starting points so that the filter doesn't fall of the image
        // These variables define the strating points for the original image
        //int start_row = (row + (2 * (kernel_size/2))) - (kernel_size/2);
        int start_row = row;
        int start_col = col;

        // For every filter in layer
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

                        if (row_val >= 0 && row_val < num_rows_img && col_val >= 0 && col_val < num_cols_img){
                            // Change the channel index 
                            //img_index = ((row_val*num_cols_img) + col_val)*num_channels + k;
                            img_index = ((row_val*num_cols_img) + col_val)*num_channels + j;
                            
                            //filter_index = (((k*kernel_size) + l)*num_channels + j)* + i;
                            filter_index = (k*kernel_size) + l + (j*kernel_size*kernel_size) + (i*num_channels*kernel_size*kernel_size);
                            
                            prod += (d_in_img[img_index] * d_layer_wt[filter_index]);
                        }
                    }
                }
            }
            img_index_out = ((row*num_cols) + col)*num_filters + i;
            // Do ReLU operation and write the output
            if (prod <= 0)
                d_layer_conv_out[img_index_out] = 0;
            else
                d_layer_conv_out[img_index_out] = prod;
        }
        }
    }
}

void Max_Pool_Kernel(float *d_layer_prev_conv_out, int stride, int max_pool_size,
                     float *d_layer_pool_out, int size_after_max_pool,
                     int num_rows_after_conv, int num_filters){
    
    for (int row=0;row<size_after_max_pool;row++){
        for (int col=0;col<size_after_max_pool;col++){
        int img_index, img_index_out, row_val, col_val;
        
        // Defining starting points so that the filter doesn't fall of the image
        int start_row = row*stride;
        int start_col = col*stride;

        // For every filter in layer
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
}

void Dense_Layer_Kernel(float *d_layer_conv_out, const float *d_layer_wt, const float *d_layer_bias,
                        float *d_pred, int in_size,
                        int num_dense_elements, int num_flattened_elements){

    for (int tid=0;tid<num_dense_elements;tid++){

        int index;
        float prod = d_layer_bias[tid];
        
        // For every filter in the previous conv layer
        for (int i=0;i<num_flattened_elements;i++){
            index = (tid*num_flattened_elements + i);
            prod += d_layer_conv_out[i] * d_layer_wt[index];
        }
        if (prod > 0)
            d_pred[tid] = prod;
        else
            d_pred[tid] = 0.0f;
    }
}

void Dense_Layer_Final_Kernel(float *d_layer_prev_out, const float *d_layer_wt,
                              float *d_final_pred, int in_size){

    for (int tid=0;tid<in_size;tid++){
        d_final_pred[tid] = d_layer_prev_out[tid] * d_layer_wt[tid];
    }
}
