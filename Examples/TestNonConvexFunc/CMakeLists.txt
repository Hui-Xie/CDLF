
set(TesNonConvexFuncSRCS NonconvexNet.cpp TestNonConvecFunc.cpp LossNonConvexExample1.cpp LossNonConvexExample2.cpp)
add_executable(TesNonConvexFunc ${TesNonConvexFuncSRCS})
target_link_libraries(TesNonConvexFunc ${CDLFLib})