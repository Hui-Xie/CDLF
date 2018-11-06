
set(TestConvolutionSRCS ConvNet.cpp TestConvolution.cpp)
add_executable(TestConvolution ${TestConvolutionSRCS})
target_link_libraries(TestConvolution ${CDLFLib} ${CUDA_LIBRARIES} pthread)