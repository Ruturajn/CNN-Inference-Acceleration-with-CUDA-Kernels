# CNN-Inference-Acceleration-with-CUDA-Kernels

This repository contains Custom CUDA Kernels, that accelerate the Inference of CNNs.

```
./
├── CD-Classfier-Script
├── CD-Classfier-Timing
├── CD-Classifier
│   ├── Data.m
│   ├── inc
│   │   ├── CNN_Funcs.h
│   │   ├── CNNWeights_Layer1.h
│   │   ├── CNNWeights_Layer2.h
│   │   ├── CNNWeights_Layer3_128.h
│   │   └── CNNWeights_Layer4_1.h
│   ├── Input_Image.txt
│   ├── Makefile
│   ├── Model_CD.m
│   ├── src
│   │   ├── CNN_Inference.cu
│   │   ├── CNN_Layers.cu
│   │   ├── main.cpp
│   │   └── Pre_Process.cu
│   └── Text-Weight-Files
│       ├── Bias_Layer_1.txt
│       ├── Bias_Layer_2.txt
│       ├── Conv_Layer_1.txt
│       ├── Conv_Layer_2.txt
│       ├── Dense_Layer_128_Bias.txt
│       ├── Dense_Layer_128_Weights.txt
│       └── Dense_Layer_1_Weights.txt
├── CD-Classifier-CPP
│   ├── Data.m
│   ├── inc
│   │   ├── CNN_Funcs.h
│   │   ├── CNNWeights_Layer1.h
│   │   ├── CNNWeights_Layer2.h
│   │   ├── CNNWeights_Layer3_128.h
│   │   └── CNNWeights_Layer4_1.h
│   ├── Input_Image.txt
│   ├── Makefile
│   ├── Model_CD.m
│   ├── src
│   │   ├── CNN_Inference.cpp
│   │   ├── CNN_Layers.cpp
│   │   ├── main.cpp
│   │   └── Pre_Process.cpp
│   └── Text-Weight-Files
│       ├── Bias_Layer_1.txt
│       ├── Bias_Layer_2.txt
│       ├── Conv_Layer_1.txt
│       ├── Conv_Layer_2.txt
│       ├── Dense_Layer_128_Bias.txt
│       ├── Dense_Layer_128_Weights.txt
│       └── Dense_Layer_1_Weights.txt
├── CD-Classifier-Speed-Up
│   ├── Data.m
│   ├── inc
│   │   ├── CNN_Funcs.h
│   │   ├── CNNWeights_Layer1.h
│   │   ├── CNNWeights_Layer2.h
│   │   ├── CNNWeights_Layer3_128.h
│   │   └── CNNWeights_Layer4_1.h
│   ├── Input_Image.txt
│   ├── Makefile
│   ├── Model_CD.m
│   ├── src
│   │   ├── CNN_Inference.cu
│   │   ├── CNN_Layers.cu
│   │   ├── main.cpp
│   │   └── Pre_Process.cu
│   └── Text-Weight-Files
│       ├── Bias_Layer_1.txt
│       ├── Bias_Layer_2.txt
│       ├── Conv_Layer_1.txt
│       ├── Conv_Layer_2.txt
│       ├── Dense_Layer_128_Bias.txt
│       ├── Dense_Layer_128_Weights.txt
│       └── Dense_Layer_1_Weights.txt
├── CNN-Accel
│   ├── inc
│   │   ├── CNN_Funcs.h
│   │   ├── CNNWeights_Layer1.h
│   │   ├── CNNWeights_Layer2.h
│   │   └── CNNWeights_Layer3.h
│   ├── Makefile
│   ├── src
│   │   ├── CNN_Inference.cu
│   │   ├── main.cpp
│   │   └── Pre_Process.cu
│   └── Text-Weight-Files
│       ├── Bias_Layer_1.txt
│       ├── Bias_Layer_2.txt
│       ├── Conv_Layer_1.txt
│       ├── Conv_Layer_2.txt
│       └── Dense_Layer_Weight.txt
├── CNN-Accel-CPP
│   ├── inc
│   │   ├── CNN_Funcs.h
│   │   ├── CNNWeights_Layer1.h
│   │   ├── CNNWeights_Layer2.h
│   │   └── CNNWeights_Layer3.h
│   ├── Makefile
│   ├── src
│   │   ├── CNN_Inference.cpp
│   │   ├── main.cpp
│   │   └── Pre_Process.cpp
│   └── Text-Weight-Files
│       ├── Bias_Layer_1.txt
│       ├── Bias_Layer_2.txt
│       ├── Conv_Layer_1.txt
│       ├── Conv_Layer_2.txt
│       └── Dense_Layer_Weight.txt
└── CNN-Accel-Test
    ├── inc
    │   ├── CNN_Funcs.h
    │   ├── CNNWeights_Layer1.h
    │   ├── CNNWeights_Layer2.h
    │   └── CNNWeights_Layer3.h
    ├── Makefile
    ├── src
    │   ├── CNN_Inference.cu
    │   ├── main.cpp
    │   └── Pre_Process.cu
    └── Text-Weight-Files
        ├── Bias_Layer_1.txt
        ├── Bias_Layer_2.txt
        ├── Conv_Layer_1.txt
        ├── Conv_Layer_2.txt
        └── Dense_Layer_Weight.txt

24 directories, 101 files
```
