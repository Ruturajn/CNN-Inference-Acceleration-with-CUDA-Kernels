#------------------------------------------------------------------------------
# A Makefile for building a CNN Inference project (CPU Version), which 
# contains .cpp files. The build files are stored in
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
CC = g++
CPP_FLAGS = -g -std=c++11

OPENCV_FLAGS = `pkg-config opencv --cflags --libs`
LIBS = -I ./inc/

#------------------------------------------------------------------------------

# Defining the CPP and CUDA source files, also naming the output target
SRC_CPP = ./src/main.cpp ./src/Pre_Process.cpp ./src/CNN_Inference.cpp
BUILD_DIR = build

TARGET = cnn

#------------------------------------------------------------------------------

# Defining pattern rule for building .o files from .cpp files
CPP_OBJS = $(notdir $(SRC_CPP:.cpp=.o))
%.o : ./src/%.cpp
	@echo "\033[1;32m[ * ] Building cpp source file ...\033[0;37m"
	$(CC) $(CPP_FLAGS) $(LIBS) $(OPENCV_FLAGS) -c $< -o ./$(BUILD_DIR)/$@

#------------------------------------------------------------------------------

#Defining the recipes and various targets
.PHONY: all clean help

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	@echo "\033[1;35m[ * ] Creating Build Directory ...\033[0;37m"
	mkdir -p $(BUILD_DIR)

$(TARGET): $(CPP_OBJS)
	@echo "\033[1;34m[ * ] Linking into Final Executable ...\033[0;37m"
	$(CC) $(wildcard ./$(BUILD_DIR)/*.o) -o ./$(BUILD_DIR)/$(TARGET) $(CPP_FLAGS) $(LIBS) $(OPENCV_FLAGS)

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

#g++ ./build/CNN_Inference.o ./build/main.o ./build/Pre_Process.o -o ./build/cnn -g -I ./inc -std=c++11 `pkg-config opencv --cflags --libs`
