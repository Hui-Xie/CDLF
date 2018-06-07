project (RL_NonConvex)

cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(BLAZE_INCLUDE_DIR  D:/blaze-3.3/) # for windows
#find_package( blaze  REQUIRED)
include_directories(${BLAZE_INCLUDE_DIR})

set(Common_SRCS  Layer.cpp FCLayer.cpp Net.cpp)
set(RL_NonConvexSRCS ${Common_SRCS} RL_NonConvex.cpp InputLayer.cpp InputLayer.h)
add_executable(RL_NonConvex ${RL_NonConvexSRCS})
#if( blaze_FOUND )
#    add_library( RL_NonConvex INTERFACE )
#    target_link_libraries( RL_NonConvex INTERFACE blaze::blaze )
# endif()


