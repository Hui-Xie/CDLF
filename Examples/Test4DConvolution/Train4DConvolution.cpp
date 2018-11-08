//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//
/*
 * before GPU implementation: this program run 15 mins in HPC in debug mode;
 *
 * */


#include "Conv4DNet.h"

int main (int argc, char *argv[])
{
    cout<<"Test 4D Convolution"<<endl;
    printCurrentLocalTime();

    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    Conv4DNet net("Conv4DNet", "/home/hxie1/temp_netParameters");

    if (isEmptyDir(net.getDir())) {
        net.build();
        net.initialize();
        net.setLearningRate(0.001);
        net.setLossTolerance(0.02);
        net.setBatchSize(20);
        net.printArchitecture();
    }
    else{
        net.load();
    }
    //  run network

    net.train();
    net.test();
    net.save();
    cout<< "=========== End of Test:  "<< net.getName() <<" ============"<<endl;
    printCurrentLocalTime();
    return 0;
}
