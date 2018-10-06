//
// Created by Hui Xie on 8/29/18.
//

#include "DAGNet.h"

int main (int argc, char *argv[])
{
#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    DAGNet net("DAGNet");
    net.build();
    net.printArchitecture();

    net.setLearningRate(0.01);
    net.setLossTolerance(0.02);
    net.setBatchSize(20);
    net.initialize();

    net.train();
    net.test();
    std::cout<<"====================End of This Program==================="<<std::endl;
    return 0;
}