project (RL_NonConvex)
cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(Common_SRCS  Layer.cpp FCLayer.cpp Net.cpp)

set(RL_NonConvexSRCS ${Common_SRCS} RL_NonConvex.cpp)
add_executable(RL_NonConvex ${RL_NonConvexSRCS})


