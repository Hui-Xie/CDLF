//
// Created by Hui Xie on 8/13/2018.
//
#include "ConvNet.h"

int main (int argc, char *argv[])
{
    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    cout<<"Notes:"<<endl;
    cout<<"This program test that 2 simple convolutional layers can approximate a convex function, and converge."<<endl;
    cout<<"This program support real 3D convolution."<<endl;

    ConvNet net("ConvNet", ".");

    if (isEmptyDir(net.getDir())) {
        net.build();
        net.initialize();
        net.setLearningRate(0.001);
        net.setLossTolerance(0.02);
        net.setBatchSize(20);
    }
    else{
        net.load();
    }
    net.printArchitecture();

    //  run network
    net.train();
    net.test();
    cout<< "=========== End of Test:  "<< net.getName() <<" ============"<<endl;
    return 0;
}