
set(TestVNetSRCS VNet.cpp TestVNet.cpp)
add_executable(TestVNet ${TestVNetSRCS})
target_link_libraries(TestVNet ${CDLFLib}   )