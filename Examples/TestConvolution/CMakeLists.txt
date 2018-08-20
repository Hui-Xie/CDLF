
set(TestConvolutionSRCS ConvNet.cpp TestConvolution.cpp LossConvexExample1.cpp)
add_executable(TestConvolution ${TestConvolutionSRCS})
target_link_libraries(TestConvolution ${CDLFLib})