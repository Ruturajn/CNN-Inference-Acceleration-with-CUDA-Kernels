#ifndef __CNN_Funcs_H__
#define __CNN_Funcs_H__

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

// Pre-processing function handling allocating memory in device
// and transfering data from host to device and back (In Pre_Process.cu)
float pre_process(Mat &in_img);

// The Convolution Layer Function that calls the Convolution Kernel
float Conv_Layer(float *in_img, const float *layer_wt, const float *layer_bias,
                 int num_filters, int num_channels, int output_size,
                 float *layer_conv_out, int in_size, int num_wt_elements,
                 int kernel_size);

// The Max Pool Function that calls the Max Pooling Kernel
float Max_Pool_Layer(float *layer_prev_conv_out, int max_pool_size,
                    int max_pool_stride, float *max_pool_out,
                    int output_size, int num_filters,
                    int in_size);

// The Dense Layer Function that calls the Dense Layer kernel
float Dense_Layer(float *layer_prev_out,const float *layer_wt, const float *layer_bias,
                 float *layer_out, int in_size, int num_filters, int out_size,
                 int num_wt_elements, int num_flattened_elements);

// The Final Dense Layer function that calls the Final Dense Layer Kernel
float Dense_Layer_Final(float *layer_prev_out, const float *layer_wt,
                        float *final_pred, int in_size, int num_wt_elements);

// Convolution Kernel (In CNN_Inference.cu)
void Conv_Kernel(float *d_in_img, const float *d_layer_wt,
                 const float *d_layer_bias, int num_rows,
                 int num_cols, float *d_layer_conv_out,
                 int kernel_size, int num_filters,
                 int num_channels);

// Max-Pooling Kernel (In CNN_Inference.cu)
void Max_Pool_Kernel(float *d_layer_prev_conv_out, int stride, int max_pool_size,
                     float *d_layer_pool_out, int size_after_max_pool,
                     int num_rows_after_conv, int num_filters);

// Dense Layer Kernel (In CNN_Inference.cu)
void Dense_Layer_Kernel(float *d_layer_conv_out, const float *d_layer_wt, const float *d_layer_bias,
                        float *d_pred, int in_size,
                        int num_dense_elements, int num_flattened_elements);

// Final Dense Layer Kernel (In CNN_Inference.cu)
void Dense_Layer_Final_Kernel(float *d_layer_prev_out, const float *d_layer_wt,
                              float *d_final_pred, int in_size);

#endif
