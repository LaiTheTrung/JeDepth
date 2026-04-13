################################################################################
# Makefile for HitNet Custom Parser for DeepStream
################################################################################

# DeepStream SDK path
DEEPSTREAM_DIR := /opt/nvidia/deepstream/deepstream-7.1

# CUDA paths
CUDA_VER ?= 12.6
CUDA_DIR := /usr/local/cuda-$(CUDA_VER)

# Compiler
CXX := g++
NVCC := $(CUDA_DIR)/bin/nvcc

# Target library
TARGET_LIB := nvdsinfer_custom_impl_hitnet.so

# Source files
SRCS := nvdsinfer_hitnet_parser.cpp

# Include paths
INCLUDES := \
    -I$(DEEPSTREAM_DIR)/sources/includes \
    -I$(CUDA_DIR)/include

# Library paths
LIB_PATHS := \
    -L$(CUDA_DIR)/lib64

# Libraries to link
# LIBS := \
#     -lnvinfer \
#     -lnvparsers \
#     -lcudart

# Compiler flags
CXXFLAGS := \
    -std=c++11 \
    -fPIC \
    -shared \
    $(INCLUDES)

# Linker flags
# LDFLAGS := \
#     $(LIB_PATHS) \
#     $(LIBS)

# Object files
OBJS := $(SRCS:.cpp=.o)

################################################################################
# Build rules
################################################################################

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $@"

%.o: %.cpp
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Cleaning..."
	rm -rf $(OBJS) $(TARGET_LIB)

.PHONY: all clean install install-local

################################################################################
# Help
################################################################################

help:
	@echo "HitNet Custom Parser Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build the custom parser library"
	@echo "  clean          - Remove built files"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_VER       - CUDA version (default: 11.8)"
	@echo "  DEEPSTREAM_DIR - DeepStream installation path"
	@echo ""
	@echo "Example:"
	@echo "  make CUDA_VER=12.0"
	@echo "  make install"