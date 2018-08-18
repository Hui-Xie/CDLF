# MNIST dataset: http://yann.lecun.com/exdb/mnist/

set (MNISTDir ../../Interfaces3ndParty/MNIST)
include_directories(${MNISTDir})

set(MnistSRCS MnistConvNet.cpp MnistTest.cpp ../../Interfaces3ndParty/MNIST/MNIST.cpp ../../Interfaces3ndParty/MNIST/MNIST.h)
add_executable(MnistTest ${MnistSRCS})
target_link_libraries(MnistTest ${CDLFLib})