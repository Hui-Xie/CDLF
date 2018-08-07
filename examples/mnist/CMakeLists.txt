# MNIST dataset: http://yann.lecun.com/exdb/mnist/

set(MnistSRCS MnistTools.cpp MnistTest.cpp)
add_executable(MnistTest ${MnistSRCS})
target_link_libraries(MnistTest ${CDLFLib})