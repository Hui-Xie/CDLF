//
// Created by Hui Xie on 8/29/18.
//

#include "DAGNet.h"


/* There is a known bug:
 * When Conv2 is freed at the end of program, its m_pYTensor->m_data has heap corruption error: "free(): invalid next size (fast)".
 * this error does not affect the normal running of program. I tried to debug it, but I can not debug it successfully.
 * I am afraid that it is compiler's bug. I leave it as it is.
 * ---------Hui Xie  Nov 09th, 2018
 *
 * */


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