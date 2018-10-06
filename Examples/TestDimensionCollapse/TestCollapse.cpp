//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "CollapseNet.h"

int main (int argc, char *argv[])
{
    cout<<"Test Dimensional Collapse in Convolution"<<endl;

#ifdef UseGPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif


    CollapseNet net("CollapseNet");
    net.build();

    // config network parameters;
    net.setLearningRate(0.001);
    net.setLossTolerance(0.02);
    net.setBatchSize(20);
    net.printArchitecture();

    //  run network
    net.initialize();
    net.train();
    net.test();
    cout<< "=========== End of Dimensional Collapse in Convolution Test ============"<<endl;
    return 0;
}
