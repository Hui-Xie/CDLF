
set(TestPerfSRCS PerfNet.cpp TestPerf.cpp)
add_executable(TestPerf ${TestPerfSRCS})
target_link_libraries(TestPerf ${CDLFLib} ${CUDA_LIBRARIES} pthread)