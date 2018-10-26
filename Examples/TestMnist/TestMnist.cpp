//
// Created by Hui Xie on 8/6/2018.
//

#include "MnistConvNet.h"
#include "MNIST.h"

void printUsage(char* argv0){
    cout<<"Test 10 digits 0-9 Simultaneously in MNIST Dataset:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" fullPathOfMnistDataDir netType"<<endl;
    cout<<"where, netType string may choose 2D or 4D, which will build different convolution networks."<<endl;
}


int main(int argc, char *argv[]){

    printCurrentLocalTime();
    
    if (3 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string mnistDir = argv[1];
    const string netType = argv[2]; // 2D or 4D

    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    // Load MNIST Data
    MNIST mnist(mnistDir);
    mnist.loadData();

    cout<<"test the matching between image and label in whole dataset"<<endl;
    srand (time(NULL));
    long index = rand() % 10000;
    mnist.displayImage(mnist.m_pTestImages, index);
    cout<<"Image is "<<(int)(mnist.m_pTestLabels->e(index))<<endl;

    //tailor data and delete wholeDataSet, and keep PartDataSet
    //mnist.tailorData();
    //mnist.deleteWholeDataSet();

    //cout<<"test the matching between image and label in part dataset"<<endl;
    //index = rand() % 10000;
    //mnist.displayImage(mnist.m_pTrainImagesPart, index);
    //cout<<"Image is "<<(int)(mnist.m_pTrainLabelsPart->e(index))<<endl;

    // Construct FeedForwardNet and Train, Test
    MnistConvNet net("MnistConvNet", &mnist);
    //net.build();
    if (netType == "2D"){
        net.build2DConvolutionNet();
    }
    else if (netType == "4D"){
        net.build4DConvolutionNet();
    }
    else{
        cout<<"Error: the netType parameter is incorrect. Program exit."<<endl;
        return -3;
    }

    net.setLearningRate(0.001);
    net.setLossTolerance(0.02);
    net.setBatchSize(100);
    net.initialize();

    net.printArchitecture();
    long epoch= 2000;
    float accuracy = 0;
    for (long i=0; i<epoch; ++i){
        net.train();
        accuracy = net.test();
        cout<<"Epoch_"<<i<<": "<<" accuracy = "<<accuracy<<endl;
     }
    cout<< "=========== End of Test:  "<<net.getName() <<" ============"<<endl;
    return 0;
}