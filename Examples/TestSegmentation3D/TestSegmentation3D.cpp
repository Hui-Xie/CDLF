//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "TestSegmentation3D.h"
#include <iostream>
#include "Segmentation3DNet.h"
#include "SegmentGNet.h"
#include "SegmentDNet.h"

#include "StubNetForD.h"
#include "DataManager.h"

using namespace std;

void printUsage(char* argv0){
    cout<<"A Generative Adversarial Network for Global 3D Medical Images Segmentation: "<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" <imageAndLabelDir> [outputTestLabelsDir]"<<endl;
    cout<<"Where the imageAndLabelDir must include 4 subdirectories: testImages  testLabels  trainImages  trainLabels" <<endl;
    cout<<"And the corresponding images file and label file should have same filename in different directory. "<<endl;
    cout<<"outputTestLabelsDir is the directory for outputing test label files"<<endl;
}

int main(int argc, char *argv[])
{
    cout<<"3D segmentation for One Sample";
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


    printUsage(argv[0]);
    if (2 != argc && 3 != argc){
        cout<<"Error: parameter error. Exit. "<<endl;
        return -1;
    }
    string dataSetDir = argv[1];
    string outputLabelsDir = "";
    if (3 == argc ){
        outputLabelsDir = argv[2];
    }
    DataManager dataMgr(dataSetDir, outputLabelsDir);

    SegmentGNet Gnet("Generative Network");
    Gnet.build();
    Gnet.setLearningRate(0.001);
    Gnet.setLossTolerance(0.02);
    Gnet.setBatchSize(3);
    Gnet.initialize();
    Gnet.printArchitecture();

    SegmentDNet Dnet("Discriminative Network");
    Dnet.build();
    Dnet.setLearningRate(0.001);
    Dnet.setLossTolerance(0.02);
    Dnet.setBatchSize(3);
    Dnet.initialize();
    Dnet.printArchitecture();

    StubNetForD stubNet("StubNetwork for Discriminative Network");
    stubNet.build();
    stubNet.setBatchSize(3);
    stubNet.initialize();
    stubNet.printArchitecture();

    Segmentation3DNet gan("3DSegmentationGAN", &Gnet,&Dnet);
    gan.setDataMgr(&dataMgr);
    gan.setStubNet(&stubNet);
    gan.setStubLayer(stubNet.getFinalLayer());

    // pretrain DNet
    cout<<"Info: start pretrain D "<<endl;
    printCurrentLocalTime();
    int epochsPretrainD = 1; //100;
    for (int i=0; i< epochsPretrainD; ++i){
        gan.pretrainD();
    }


    // train G, D: quick alternative train
    cout<<"Info: start quickly switch train G and D "<<endl;
    printCurrentLocalTime();
    int epochsQuickSwitch = 1; //100;
    for (int i=0; i<epochsQuickSwitch; ++i){
        gan.quicklySwitchTrainG_D();
    }
    // train G, D: slowly alternative train
    cout<<"Info: start slowly switch train G and D "<<endl;
    printCurrentLocalTime();
    int epochsSlowSwitch = 1;//100;
    int epochsAlone = 1;// 20;
    for (int i=0; i< epochsSlowSwitch; ++i){
        for(int j=0; j< epochsAlone; ++j){
            gan.trainD();
        }
        for(int j=0; j<epochsAlone; ++j){
            gan.trainG();
        }

        cout<<"Slowly Switch Epoch: "<<i<<endl;
        printCurrentLocalTime();
        if (i != epochsSlowSwitch -1){
            gan.testG(false);
        }
        else{
            gan.testG(true); //output file
        }

    }

    cout<< "=========== End of Test:  "<<gan.getName() <<" ============"<<endl;
    printCurrentLocalTime();
    return 0;
}