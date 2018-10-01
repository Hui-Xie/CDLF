
set(TestCollapseSRCS CollapseNet.cpp TestCollapse.cpp LossConvexExample1.cpp)
add_executable(TestCollapse ${TestCollapseSRCS})
target_link_libraries(TestCollapse ${CDLFLib} ${CUDA_LIBRARIES})