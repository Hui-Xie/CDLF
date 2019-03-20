//
// Created by Hui Xie on 12/11/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#include "HNSegVNet.h"

void printUsage(char* argv0){
    cout<<"Train Head&Neck Suqamous Cell Carcinoma Segmentation V Model:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> <fullPathOfRadiomicsDataDir>  <learningRate>"<<endl;
    cout<<"For examples: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters/CATonsil /home/hxie1/data/HeadNeckSCC/CATonsil  0.005"<<endl;
    cout<<argv0<<" /Users/hxie1/temp_netParameters/CATonsil  /Users/hxie1/data/HeadNeckSCC/CATonsil 0.005"<<endl;
}


int main(int argc, char *argv[]){
    printCurrentLocalTime();
    if (4 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string netDir = argv[1];
    const string dataDir = argv[2];
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

#ifdef Use_GPU
    HNSegVNet net(netDir);
#else
    HNSegVNet net(netDir);
#endif

    AdamOptimizer adamOptimizer(0.001,0.9,0.999);
    net.setOptimizer(&adamOptimizer);

    cout<<"=========================================="<<endl;
    cout<<"Info: this is "<<net.getName() <<" net."<<endl;
    cout<<"=========================================="<<endl;

    if (!isEmptyDir(net.getDir())) {
        net.load();
    }
    else{
        cout<<"Error: program can not load net."<<endl;
        return -2;
    }
    //net.defineAssemblyLoss();
    net.printArchitecture();
    //net.setLearningRate(learningRate);
    //net.initializeLRs(learningRate);
    net.detectSoftmaxBeforeLoss();
    net.allocateOptimizerMem("Adam");

    //for one sample training
    //net.setOneSampleTrain(true);

    HNDataManager dataMgr(dataDir);
    net.m_pDataMgr = &dataMgr;

    int epoch= 15000;
    for (int i=0; i<epoch; ++i){
        cout <<"Epoch "<<i<<": "<<endl;
        net.train();
        if (0 == (i+1)%5 ){ // 18min/epoch, about 1.5 hour/saving
            net.save();
        }

        net.test();

        /*
        //decay learning rate
        if ( 0 == (i+1)%10){
            float R = net.getLearningRate();
            float newR = R*0.6;
            if (newR < 1e-7){
                newR = 1e-7;
            }
            net.setLearningRate(newR);
        }
        */

        if (net.getOneSampleTrain()) break;

    }
    cout<< "=========== End of Test:  "<<net.getName() <<" ============"<<endl;

    return 0;
}