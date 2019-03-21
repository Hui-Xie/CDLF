
set(Train4DConvolutionSRCS Conv4DNet.cpp Train4DConvolution.cpp)
add_executable(Train4DConvolution ${Train4DConvolutionSRCS})
target_link_libraries(Train4DConvolution ${CDLFLib}   )