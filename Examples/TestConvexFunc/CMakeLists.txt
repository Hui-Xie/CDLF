

set(TestConvexFuncSRCS ConvexNet.cpp TestConvexFunc.cpp)
add_executable(TestConvexFunc ${TestConvexFuncSRCS})
target_link_libraries(TestConvexFunc ${CDLFLib} ${CUDA_LIBRARIES} pthread)