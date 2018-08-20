# MNIST dataset: http://yann.lecun.com/exdb/mnist/

set (MNISTDir ../../Interface3rdParty/MNIST)
include_directories(${MNISTDir})

set(TestMnistSRCS MnistConvNet.cpp TestMnist.cpp ../../Interface3rdParty/MNIST/MNIST.cpp ../../Interface3rdParty/MNIST/MNIST.h)
add_executable(TestMnist ${TestMnistSRCS})
target_link_libraries(TestMnist ${CDLFLib})