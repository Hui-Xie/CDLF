//
// Created by Hui Xie on 8/13/2018.
//
#include "ConvNet.h"

int main (int argc, char *argv[])
{
#ifdef UseGPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    cout<<"Notes:"<<endl;
    cout<<"This program test that 2 simple convolutional layers can approximate a convex function, and converge."<<endl;
    cout<<"This program support real 3D convolution."<<endl;

    ConvNet net("ConvNet");
    net.build();
    net.printArchitecture();

    // config network parameters;
    net.setLearningRate(0.001);
    net.setLossTolerance(0.02);
    net.setBatchSize(20);

    //  run network
    net.initialize();
    net.train();
    net.test();
    cout<< "=========== End of ConvolutionLayer Test ============"<<endl;
    return 0;
}