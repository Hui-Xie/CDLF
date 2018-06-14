project (RL_NonConvex)

cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# https://bitbucket.org/blaze-lib/blaze
set(BLAZE_INCLUDE_DIR  D:/blaze-3.3/) # for windows
include_directories(${BLAZE_INCLUDE_DIR})

# https://github.com/kthohr/stats
set(STATS_INCLUDE_DIR  E:/CProject/stats/include/) # for windows  stats C++ statistics library
include_directories(${STATS_INCLUDE_DIR})

set(Common_SRCS  Layer.cpp FCLayer.cpp InputLayer.cpp LossLayer.cpp Net.cpp statisTool.cpp ReLU.cpp NormalizationLayer.cpp)
set(RL_NonConvexSRCS ${Common_SRCS} RL_NonConvex.cpp LossConvexExample1.cpp LossConvexExample1.h)
add_executable(RL_NonConvex ${RL_NonConvexSRCS})
#if( blaze_FOUND )
#    add_library( RL_NonConvex INTERFACE )
#    target_link_libraries( RL_NonConvex INTERFACE blaze::blaze )
# endif()


