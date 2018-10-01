
set(Test4DConvolutionSRCS Conv4DNet.cpp Test4DConvolution.cpp LossConvexExample1.cpp)
add_executable(Test4DConvolution ${Test4DConvolutionSRCS})
target_link_libraries(Test4DConvolution ${CDLFLib} ${CUDA_LIBRARIES})