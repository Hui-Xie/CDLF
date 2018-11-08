//
// Created by Hui Xie on 8/29/18.
//

#include "DAGNet.h"

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

    DAGNet net("DAGNet", "/home/hxie1/temp_netParameters");

    if (isEmptyDir(net.getDir())) {
        net.build();
        //net.buildSimple();
        net.initialize();
        net.setLearningRate(0.01);
        net.setLossTolerance(0.02);
        net.setBatchSize(20);
        net.setEpoch(100);
    }
    else{
        net.load();
    }
    net.printArchitecture();


    net.train();
    net.test();
    net.save();
    cout<< "=========== End of Test:  "<<net.getName() <<" ============"<<endl;
    return 0;
}