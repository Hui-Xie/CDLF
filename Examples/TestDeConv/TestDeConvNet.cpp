//
// Created by Hui Xie on 01/15/19.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#include "DeConvNet.h"

void printUsage(char* argv0){
    cout<<"Test DeConv Network:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> <learningRate>"<<endl;
    cout<<"For examples: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters 0.1"<<endl;
}


int main(int argc, char *argv[]){
    printCurrentLocalTime();
    if (3 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string netDir = argv[1];
    const float learningRate = stof(argv[2]);

    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    //Load MnistAutoEncoder Net
    DeConvNet net(netDir);
    if (!isEmptyDir(net.getDir())) {
        net.load();
    }
    else{
        cout<<"Error: program can not load net."<<endl;
        return -2;
    }
    net.printArchitecture();
    net.setLearningRate(learningRate);
    net.setUnlearningLayerID(10);

    //net.setOneSampleTrain(true);

    int epoch= 1500000;
    for (int i=0; i<epoch; ++i){
        net.train();

        if (0 == (i+1)%23){
            net.save();
        }        
        if (! net.getOneSampleTrain()){
            net.test();
        }
        cout<<"Epoch_"<<i<<": "<<" mean squareLoss for each element = "<< net.m_loss <<endl;
    }
    cout<< "=========== End of Test:  "<<net.getName() <<" ============"<<endl;
    return 0;
}