
set(TestDAGSRCS DAGNet.cpp TestDAGNet.cpp)
add_executable(TestDAGNet ${TestDAGSRCS})
target_link_libraries(TestDAGNet ${CDLFLib} ${CUDA_LIBRARIES} )