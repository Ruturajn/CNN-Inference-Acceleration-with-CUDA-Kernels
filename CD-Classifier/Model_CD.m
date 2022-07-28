%% Import the Weight File
% clc
% clear
% close all;
net = importKerasNetwork("dogcat_model_bak.h5")
%place_layers = findPlaceholderLayers(net)
% plot(net)
net.Layers

%% Do the convolution for the first layer

% in_img = imread('20.jpg');
% double_in_img = double(in_img);
% norm_in_img = double_in_img .* (1/255);
% resized_in_img = imresize(norm_in_img, [64 64]);
pad = net.Layers(2,1).PaddingSize;
stride = net.Layers(2,1).Stride;
wts = net.Layers(2,1).Weights;
bias = net.Layers(2,1).Bias;
% 
% new_opencv_img(:,:,1) = resized_in_img(:,:,3);
% new_opencv_img(:,:,2) = resized_in_img(:,:,2);
% new_opencv_img(:,:,3) = resized_in_img(:,:,1);

% in_img_id = fopen("Input_Image.txt",'r');
% for k=1:3
%     for i=1:64
%         for j=1:64
%             new_opencv_img(i,j,k) = textscan(in_img_id, '%f %f %f %f %f %f %f % f %f %f %f %f %f %f %f %f' ...
%                 '%f %f %f %f %f %f %f % f %f %f %f %f %f %f %f %f' ...
%                 '%f %f %f %f %f %f %f % f %f %f %f %f %f %f %f %f' ...
%                 '%f %f %f %f %f %f %f % f %f %f %f %f %f %f %f %f');
%             tmp = fscanf(in_img_id,'\n');
%         end
%         tmp = fscanf(in_img_id,"\n");
%     end
%     tmp = fscanf(in_img_id,"\n");
% end
% fclose(in_img_id);

Layer_1_conv = conv_infer(new_opencv_img, wts, bias, pad(2), stride, "both");

% Layer_1_img = mat2gray(Layer_1_conv(:,:,1));
% imshow(Layer_1_img);

for k=1:size(Layer_1_conv,3)
    for j=1:size(Layer_1_conv,2)
        for i=1:size(Layer_1_conv,1)
            if (Layer_1_conv(i,j,k) < 0)
                Layer_1_conv(i,j,k) = 0;
            end
        end
    end
end

%% Max Pool Layer

Layer_1_pool = pool_infer(Layer_1_conv, [2 2],[2 2]);


%% Second Conv Layer

pad_2 = net.Layers(5,1).PaddingSize;
stride_2 = net.Layers(5,1).Stride;
wts_2 = net.Layers(5,1).Weights;
bias_2 = net.Layers(5,1).Bias;

Layer_2_conv = conv_infer(Layer_1_pool, wts_2, bias_2, pad_2(2), stride_2,"both");

for k=1:size(Layer_2_conv,3)
    for j=1:size(Layer_2_conv,2)
        for i=1:size(Layer_2_conv,1)
            if (Layer_2_conv(i,j,k) < 0)
                Layer_2_conv(i,j,k) = 0;
            end
        end
    end
end

%% Max Pooling
Layer_2_pool = pool_infer(Layer_2_conv, [2 2], [2 2]);

%% Dense Layer 1
flattened_arr = pagectranspose(Layer_2_pool);
flattened_arr = flattened_arr(:);

wts_3 = net.Layers(9,1).Weights;
bias_3 = net.Layers(9,1).Bias;

dense_layer1_out = eye([128 1]);

for i=1:size(wts_3,1)
    for j=1:size(flattened_arr,1)
        dense_layer1_out(i) = dense_layer1_out(i) + (flattened_arr(j) * wts_3(i,j));
    end
    dense_layer1_out(i) = dense_layer1_out(i) + bias_3(i);
    if (dense_layer1_out(i) < 0)
        dense_layer1_out(i) = 0;
    end
end

%% Dense Layer 2
wts_4 = net.Layers(11,1).Weights;
bias_4 = net.Layers(11,1).Bias;

prod = [bias_4];

for i=1:size(wts_4,2)
    prod(1) = prod(1) + (dense_layer1_out(i) * wts_4(1,i));
end

pred = (1 / (1 + exp(-1 * prod)));

 





