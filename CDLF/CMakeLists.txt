project (CDLF)  # C++ Deep Learning Framework

cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# https://github.com/kthohr/stats
#set(STATS_INCLUDE_DIR  E:/CProject/stats/include/) # for windows  stats C++ statistics library
set(STATS_INCLUDE_DIR /Users/hxie1/Projects/stats/include)  #for HPC
include_directories(${STATS_INCLUDE_DIR})

set(CDLF_SRCS  Layer.cpp FCLayer.cpp InputLayer.cpp LossLayer.cpp Net.cpp
        statisTool.cpp ReLU.cpp NormalizationLayer.cpp Tensor.h Tools.cpp
        ConvolutionLayer.cpp BiasLayer.cpp SoftMaxLayer.cpp MaxPoolingLayer.cpp
        IdentityLayer.cpp SumLayer.cpp BranchLayer.cpp
        SigmoidLayer.cpp  CrossEntropyLoss.cpp VectorizationLayer.cpp)

add_library(CDLF SHARED ${CDLF_SRCS})




