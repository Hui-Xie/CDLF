project (CDLF LANGUAGES CXX CUDA)  # C++ Deep Learning Framework

cmake_minimum_required (VERSION 3.8)

message("Config StatsLab Notes:")
message("0  Courtesy of Keith O'Hara (https://github.com/kthohr/stats) for StatsLab")
message("1  download code from: https://github.com/kthohr/stats into your intall directory")
message("2  in Clion File/Setting/Build/CMake/CMakeOptions config -DSTATS_INCLUDE_DIR=The/Full/Path/Of/Stats/Include/Dir ")
message("*************************************")

#set(STATS_INCLUDE_DIR /Users/hxie1/Projects/stats/include)
include_directories(${STATS_INCLUDE_DIR})

#[[
# For Cuda Library
message("Config CUDA Notes:")
message("0  Courtesy of Nvidia Company (https://www.nvidia.com/en-us/) for its cuda_toolkit;")
message("1  download from https://developer.nvidia.com/cuda-downloads into your intall directory;")
message("2  decompress, ccmake, make ITK version using same version compilers with CDLF; Different compiler will result link errors;")
# make sure ITK and CDLF to use same compiler to compile
message("3  in Clion File/Setting/Build/CMake/CMakeOptions config -DITK_DIR=the/Full/Path/Of/ITK-Build/Dir ;")
message("*************************************")


set(CUDA_HOME /opt/apps/cuda/9.1.85.2)

find_package(CUDA QUIET)
if(CUDA_FOUND)
    # Pass options to NVCC
    set(CUDA_NVCC_FLAGS   ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_22,code=sm_22)

    # Specify include directories
    include_directories(kernels  utility)
else(CUDA_FOUND )
    message("Infor: we can not find CUDA.")
endif(CUDA_FOUND)
]]




set(CDLF_SRCS  Layer.cpp FCLayer.cpp InputLayer.cpp LossLayer.cpp FeedForwardNet.cpp
        statisTool.cpp ReLU.cpp NormalizationLayer.cpp Tensor.h Tools.cpp
        ConvolutionLayer.cpp BiasLayer.cpp SoftmaxLayer.cpp MaxPoolingLayer.cpp
        IdentityLayer.cpp MergerLayer.cpp BranchLayer.cpp
        SigmoidLayer.cpp  CrossEntropyLoss.cpp VectorizationLayer.cpp ExponentialLayer.cpp
        SubTensorLayer.cpp GAN.cpp Net.cpp GNet.cpp DNet.cpp ScaleLayer.cpp)

add_library(CDLF SHARED ${CDLF_SRCS})
set_target_properties(CDLF PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

