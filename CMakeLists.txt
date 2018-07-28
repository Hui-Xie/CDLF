project (RL_NonConvex)

cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# https://bitbucket.org/blaze-lib/blaze
#set(BLAZE_INCLUDE_DIR  D:/blaze-3.3/) # for windows
#include_directories(${BLAZE_INCLUDE_DIR})

# https://github.com/kthohr/stats
set(STATS_INCLUDE_DIR  E:/CProject/stats/include/) # for windows  stats C++ statistics library
include_directories(${STATS_INCLUDE_DIR})

set(Common_SRCS  Layer.cpp FCLayer.cpp InputLayer.cpp LossLayer.cpp Net.cpp
                  statisTool.cpp ReLU.cpp NormalizationLayer.cpp Tensor.h Tools.cpp ConvolutionLayer.cpp ConvolutionLayer.h BiasLayer.cpp BiasLayer.h SoftMaxLayer.cpp SoftMaxLayer.h MaxPoolingLayer.cpp MaxPoolingLayer.h)
set(RL_NonConvexSRCS ${Common_SRCS}
                  RL_Test.cpp LossConvexExample1.cpp LossNonConvexExample1.cpp
                  LossNonConvexExample1.h LossNonConvexExample2.cpp LossNonConvexExample2.h LossConvexExample2.cpp LossConvexExample2.h)
add_executable(RL_NonConvex ${RL_NonConvexSRCS})


# for test Tensor class
set(Test_Tensor Tensor.h Tools.cpp testTensor.cpp)
add_executable(testTensor ${Test_Tensor})



