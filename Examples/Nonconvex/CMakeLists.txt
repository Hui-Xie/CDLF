
set(NonConvexSRCS NonconvexNet.cpp nonconvexTest.cpp LossNonConvexExample1.cpp LossNonConvexExample2.cpp)
add_executable(NonConvexTest ${NonConvexSRCS})
target_link_libraries(NonConvexTest ${CDLFLib})