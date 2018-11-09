//
// Created by Hui Xie on 11/9/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "MNIST.h"
#include "MnistConvNet.h"

void printUsage(char* argv0){
    cout<<"Generate Advesarial Samples for MNIST Dataset:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> <fullPathOfMnistDataDir>  <fullPathAdvData>"<<endl;
    cout<<"for examples: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters /home/hxie1/Projects/mnist  /home/hxie1/temp_advData"<<endl;
}


int main(int argc, char *argv[]){
    printCurrentLocalTime();
    if (4 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string netDir = argv[1];
    const string mnistDir = argv[2];
    string advDataDir = argv[3];

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
    bool onlyTestSet = true;
    MNIST mnist(mnistDir, onlyTestSet);
    mnist.loadData();

    //Load Mnist Net
    MnistConvNet net("MnistNet", netDir, &mnist);
    if (!isEmptyDir(net.getDir())) {
        net.load();
    }
    else{
        cout<<"Error: program can not load trained Mnist net."<<endl;
        return -2;
    }
    net.printArchitecture();

    //create Adversarial data directory
    advDataDir +=getCurTimeStr();
    createDir(advDataDir);

    // choose an initial digit file
    srand (time(NULL));
    long index = rand() % 10000;
    mnist.displayImage(mnist.m_pTestImages, index);
    cout<<"Image is "<<(int)(mnist.m_pTestLabels->e(index))<<endl;


    cout<< "=========== End of Test:  "<<net.getName() <<" ============"<<endl;
    return 0;
}