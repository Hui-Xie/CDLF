
# for test Tensor class
set(TestTensorSRCS testTensor.cpp)
add_executable(TestTensor ${TestTensorSRCS})
target_link_libraries(TestTensor ${CDLFLib} ${CUDA_LIBRARIES} pthread)
