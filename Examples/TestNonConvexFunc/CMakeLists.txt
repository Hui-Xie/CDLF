
set(TesNonConvexFuncSRCS NonconvexNet.cpp TestNonConvecFunc.cpp)
add_executable(TesNonConvexFunc ${TesNonConvexFuncSRCS})
target_link_libraries(TesNonConvexFunc ${CDLFLib}   )