
set(ConvolutionSRCS ConvolutionTest.cpp LossConvexExample1.cpp)
add_executable(ConvolutionTest ${ConvolutionSRCS})
target_link_libraries(ConvolutionTest ${CDLFLib})