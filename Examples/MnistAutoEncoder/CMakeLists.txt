
message("Config MNIST Notes:")
message("0  Courtesy of Yann LeCun (http://yann.lecun.com/exdb/mnist/) for National Institute of Standards and Technology database;")
message("1  download 4 ubyte.gz files from:http://yann.lecun.com/exdb/mnist/ into your MNIST directory;")
message("2  modify the mnistDir const string in your TestMnist.cpp file in this directory;")
message("*************************************")


set (MNISTDir ../../Interface3rdParty/MNIST)
include_directories(${MNISTDir})

set(TrainMnistAutoEncoderSRC TrainMnistAutoEncoder.cpp ${MNISTDir}/MNIST.cpp MnistAutoEncoder.cpp)
add_executable(TrainMnistAutoEncoder ${TrainMnistAutoEncoderSRC})
target_link_libraries(TrainMnistAutoEncoder ${CDLFLib} ${CUDA_LIBRARIES} pthread)


set(TestMnistAutoEncoderSRC TestMnistAutoEncoder.cpp ${MNISTDir}/MNIST.cpp MnistAutoEncoder.cpp)
add_executable(TestMnistAutoEncoder ${TestMnistAutoEncoderSRC})
target_link_libraries(TestMnistAutoEncoder ${CDLFLib} ${CUDA_LIBRARIES} pthread )