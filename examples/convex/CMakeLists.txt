

set(ConvexSRCS convexTest.cpp LossConvexExample1.cpp LossConvexExample2.cpp)
add_executable(ConvexTest ${ConvexSRCS})
target_link_libraries(ConvexTest ${CDLFLib})