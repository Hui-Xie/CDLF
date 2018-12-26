

set(TestUnitFilterSrc UnitFilterNet.cpp TestUnitFilter.cpp)
add_executable(TestUnitFilter ${TestUnitFilterSrc})
target_link_libraries(TestUnitFilter ${CDLFLib} ${CUDA_LIBRARIES} )