project (CDLF)  # C++ Deep Learning Framework

cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# https://github.com/kthohr/stats
set(STATS_INCLUDE_DIR  E:/CProject/stats/include/) # for windows  stats C++ statistics library
include_directories(${STATS_INCLUDE_DIR})

set(CDLF_SRCS  Layer.cpp FCLayer.cpp InputLayer.cpp LossLayer.cpp Net.cpp
        statisTool.cpp ReLU.cpp NormalizationLayer.cpp Tensor.h Tools.cpp
        ConvolutionLayer.cpp ConvolutionLayer.h BiasLayer.cpp BiasLayer.h
        SoftMaxLayer.cpp SoftMaxLayer.h MaxPoolingLayer.cpp MaxPoolingLayer.h
        IdentityLayer.cpp IdentityLayer.h SumLayer.cpp SumLayer.h BranchLayer.cpp
        BranchLayer.h SigmoidLayer.cpp SigmoidLayer.h CrossEntropyLoss.cpp CrossEntropyLoss.h VectorizationLayer.cpp VectorizationLayer.h)

add_library(CDLF SHARED ${CDLF_SRCS})




