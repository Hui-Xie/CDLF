

set(TestUnitFilterSrc UnitFilterNet.cpp TestUnitFilter.cpp LossConvexExample1.cpp )
add_executable(TestUnitFilter ${TestUnitFilterSrc})
target_link_libraries(TestUnitFilter ${CDLFLib})