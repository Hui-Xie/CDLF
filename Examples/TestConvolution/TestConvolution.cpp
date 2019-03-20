//
// Created by Hui Xie on 8/13/2018.
//
#include "ConvNet.h"

void printUsage(char* argv0){
    cout<<"Train Convolution Network:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> "<<endl;
    cout<<"For examples: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters/ConvNet "<<endl;
    cout<<argv0<<" /Users/hxie1/temp_netParameters/ConvNet  "<<endl;
}


int main (int argc, char *argv[])
{
    printCurrentLocalTime();
    if (2 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string netDir = argv[1];

    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    ConvNet net(netDir);

    AdamOptimizer adamOptimizer(0.001,0.9,0.999);
    net.setOptimizer(&adamOptimizer);

    net.load();

    //test a special network.
    //net.build();
    //net.setLearningRate(0.0001);
    //net.setBatchSize(20);
    //net.setEpoch(1000);


    net.printArchitecture();

    //  run network
    net.train();
    net.test();
    net.save();
    cout<< "=========== End of Test:  "<< net.getName() <<" ============"<<endl;
    return 0;
}