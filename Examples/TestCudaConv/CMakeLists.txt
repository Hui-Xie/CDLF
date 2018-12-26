
set(TestConvCudaSrc ConvCudaNet.cpp TestConvCuda.cpp)
add_executable(TestConvCuda ${TestConvCudaSrc})
target_link_libraries(TestConvCuda ${CDLFLib} ${CUDA_LIBRARIES} )