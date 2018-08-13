
set(ConvolutionSRCS ConvolutionTest.cpp)
add_executable(ConvolutionTest ${ConvolutionSRCS})
target_link_libraries(ConvolutionTest ${CDLFLib})