

set(Conv4DSRCS Conv4DNet.cpp Conv4DTest.cpp LossConvexExample1.cpp)
add_executable(Conv4DTest ${Conv4DSRCS})
target_link_libraries(Conv4DTest ${CDLFLib})