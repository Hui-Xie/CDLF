

set(TrainConvexFuncSRCS ConvexNet.cpp TrainConvexFunc.cpp)
add_executable(TrainConvexFunc ${TrainConvexFuncSRCS})
target_link_libraries(TrainConvexFunc ${CDLFLib} ${CUDA_LIBRARIES} )