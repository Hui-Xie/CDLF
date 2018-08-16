

# for test Tensor class
set(Tensor_SRCS testTensor.cpp)
add_executable(testTensor ${Tensor_SRCS})
target_link_libraries(testTensor ${CDLFLib})