# MNIST dataset: http://yann.lecun.com/exdb/mnist/


message("Config MNIST Notes:")
message("0  Courtesy of Yann LeCun (http://yann.lecun.com/exdb/mnist/) for National Institute of Standards and Technology database;")
message("1  download 4 ubyte.gz files from:http://yann.lecun.com/exdb/mnist/ into your MNIST directory;")
message("2  modify the mnistDir const string in your TestMnist.cpp file in this directory;")
message("*************************************")


set (MNISTDir ../../Interface3rdParty/MNIST)
include_directories(${MNISTDir})

set(TestMnistSRCS MnistConvNet.cpp TestMnist.cpp ../../Interface3rdParty/MNIST/MNIST.cpp ../../Interface3rdParty/MNIST/MNIST.h)
add_executable(TestMnist ${TestMnistSRCS})
target_link_libraries(TestMnist ${CDLFLib} ${CUDA_LIBRARIES} )