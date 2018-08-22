project (CDLF)  # C++ Deep Learning Framework

cmake_minimum_required (VERSION 3.8)

message("Config StatsLab Notes:")
message("0  Courtesy of Keith O'Hara (https://github.com/kthohr/stats) for StatsLab")
message("1  download code from: https://github.com/kthohr/stats into your intall directory")
message("2  in Clion File/Setting/Build/CMake/CMakeOptions config -DSTATS_INCLUDE_DIR=The/Full/Path/Of/Stats/Include/Dir ")
message("*************************************")

include_directories(${STATS_INCLUDE_DIR})

set(CDLF_SRCS  Layer.cpp FCLayer.cpp InputLayer.cpp LossLayer.cpp Net.cpp
        statisTool.cpp ReLU.cpp NormalizationLayer.cpp Tensor.h Tools.cpp
        ConvolutionLayer.cpp BiasLayer.cpp SoftMaxLayer.cpp MaxPoolingLayer.cpp
        IdentityLayer.cpp SumLayer.cpp BranchLayer.cpp
        SigmoidLayer.cpp  CrossEntropyLoss.cpp VectorizationLayer.cpp)

add_library(CDLF SHARED ${CDLF_SRCS})




