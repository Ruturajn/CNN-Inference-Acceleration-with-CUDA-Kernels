#ifndef __CNN_Funcs_H__
#define __CNN_Funcs_H__

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

// Pre-processing function handling allocating memory in device
// and transfering data from host to device and back (In Pre_Process.cu)
float pre_process(Mat &in_img);

// Convolution Kernel (In CNN_Inference.cu)
__global__ void Conv_Kernel(float *d_in_img, float *d_layer_wt,
                            float *d_layer_bias, int num_rows,
                            int num_cols, float *d_layer_conv_out,
                            int kernel_size, int num_filters,
                            int num_channels);

// Max-Pooling Kernel (In CNN_Inference.cu)
__global__ void Max_Pool_Kernel(float *d_layer_prev_conv_out, int stride, int max_pool_size,
                                float *d_layer_pool_out, int size_after_max_pool,
                                int num_rows_after_conv, int num_filters);

// Dense Layer Kernel (In CNN_Inference.cu)
__global__ void Dense_Layer_Kernel(float *d_layer_conv_out, float *d_layer_wt,
                                   float *d_final_pred,
                                   int size_after_max_pool, int num_filters);

#endif