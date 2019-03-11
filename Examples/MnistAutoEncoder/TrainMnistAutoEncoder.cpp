//
// Created by Hui Xie on 12/11/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "MnistAutoEncoder.h"

void printUsage(char* argv0){
    cout<<"Train MNIST Dataset AutoEncoder:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> <fullPathOfMnistDataDir>  learningRate"<<endl;
    cout<<"For examples: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters /home/hxie1/Projects/mnist 0.001"<<endl;
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
    const float learningRate = stof(argv[3]);

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
    bool onlyTestSet = false;
    MNIST mnist(mnistDir, onlyTestSet);
    mnist.loadData();

    //Load MnistAutoEncoder Net
    MnistAutoEncoder net(netDir, &mnist);
    if (!isEmptyDir(net.getDir())) {
        net.load();  //at Dec 17th,2018, the trained MnistNet has an accuracy of 97.26%
    }
    else{
        cout<<"Error: program can not load a trained Mnist net."<<endl;
        return -2;
    }
    net.printArchitecture();
    net.setLearningRate(learningRate);
    net.setUnlearningLayerID(100);  // 15 and 18 is the FC layers behind the Softmax of original G net.

    int epoch= 15000;
    float squareLoss = 0.0;
    for (int i=0; i<epoch; ++i){
        net.train();

        //debug
        //net.saveYTensor();
        //net.savedYTensor();

        net.save();
        squareLoss = net.test();
        cout<<"Epoch_"<<i<<": "<<" mean squareLoss for each pixel = "<< squareLoss <<endl;

        //debug
        //break;
    }
    cout<< "=========== End of Test:  "<<net.getName() <<" ============"<<endl;
    return 0;
}