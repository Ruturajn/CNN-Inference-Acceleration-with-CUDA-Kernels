#------------------------------------------------------------------------------
# A Makefile for building a CNN Inference CUDA project, which 
# contains both .cpp and .cu files. The build files are stored in
# a separate directory called "build".
# 
# Usage: make
# 
# Build Targets:
# 		all      : Builds all the source files and creates an executable.
# 		clean    : Removes all the files created by the build process.
#
# Author : Ruturaj A. Nanoti
# Date   : 15 April, 2022 
#------------------------------------------------------------------------------

# Defining Compiler and related flags
CC = nvcc
CPP_CC = g++
CPP_FLAGS = -g
NVCC_FLAGS = -G -g -w
NVCC_COMPILE_FLAGS = -x cu
ARCH_FLAGS = -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets

OPENCV_FLAGS = `pkg-config opencv --cflags --libs`
LIBS = -I ./inc/

#------------------------------------------------------------------------------

# Defining the CPP and CUDA source files, also naming the output target
SRC_CPP = ./src/main.cpp 
SRC_CU = ./src/Pre_Process.cu ./src/CNN_Inference.cu
BUILD_DIR = build

TARGET = cnn
#------------------------------------------------------------------------------

# Bold
BBlack="\033[0;30m"       # Black
BRed="\033[0;31m"         # Red
BGreen="\033[0;32m"       # Green
BYellow="\033[0;33m"      # Yellow
BBlue="\033[0;34m"        # Blue
BPurple="\033[0;35m"      # Purple
BCyan="\033[0;36m"        # Cyan
BWhite="\033[0;37m"       # White

#------------------------------------------------------------------------------

# Defining pattern rule for building .o files from .cpp files
CPP_OBJS = $(notdir $(SRC_CPP:.cpp=.o))
%.o : ./src/%.cpp
	@echo "\033[1;32m[ * ] Building cpp source file ...\033[0;37m"
	$(CPP_CC) $(CPP_FLAGS) $(LIBS) $(OPENCV_FLAGS) -c $< -o ./$(BUILD_DIR)/$@


#------------------------------------------------------------------------------

# Defining pattern rule for building .o files from .cu files
CU_OBJS = $(notdir $(SRC_CU:.cu=.o))
%.o : ./src/%.cu
	@echo "\033[1;33m[ * ] Building cu source file ...\033[0;37m"
	$(CC) $(NVCC_FLAGS) $(NVCC_COMPILE_FLAGS) $(ARCH_FLAGS) $(LIBS) $(OPENCV_FLAGS) -dc $< -o ./$(BUILD_DIR)/$@

#------------------------------------------------------------------------------

#Defining the recipes and various targets
.PHONY: all clean run help

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	@echo "\033[1;35m[ * ] Creating Build Directory ...\033[0;37m"
	mkdir -p $(BUILD_DIR)

$(TARGET): $(CU_OBJS) $(CPP_OBJS)
	@echo "\033[1;34m[ * ] Linking into Final Executable ...\033[0;37m"
	$(CC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(LIBS) $(OPENCV_FLAGS) $(wildcard ./$(BUILD_DIR)/*.o) -o ./$(BUILD_DIR)/$(TARGET)

clean:
	@echo "\033[1;34m[ * ] Removing Build Files ...\033[0;37m"
	rm -r $(BUILD_DIR)

help:
	@echo "----------------------------------------------------------------"
	@echo "The Following are valid targets for this Makefile :"
	@echo "- all   : Builds all the source files and creates an executable."
	@echo "- clean : Removes all the files created by the build process."
	@echo ""
	@echo "For Running the executable : "
	@echo "./build/cnn <Path_to_Test_Image>"
	@echo "----------------------------------------------------------------"

#------------------------------------------------------------------------------
