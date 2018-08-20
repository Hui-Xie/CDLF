

set(TestConvexFuncSRCS ConvexNet.cpp TestConvexFunc.cpp LossConvexExample1.cpp LossConvexExample2.cpp)
add_executable(TestConvexFunc ${TestConvexFuncSRCS})
target_link_libraries(TestConvexFunc ${CDLFLib})