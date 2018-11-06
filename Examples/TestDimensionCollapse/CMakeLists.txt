
set(TestCollapseSRCS CollapseNet.cpp TestCollapse.cpp)
add_executable(TestCollapse ${TestCollapseSRCS})
target_link_libraries(TestCollapse ${CDLFLib} ${CUDA_LIBRARIES} pthread)