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
    cout<<argv0<<" <imageAndLabelDir>"<<endl;
    cout<<"Where the imageAndLabelDir must include 4 subdirectories: testImages  testLabels  trainImages  trainLabels" <<endl;
    cout<<"And the corresponding images file and label file should have same filename in different directory. "<<endl;
}

int main(int argc, char *argv[])
{
    printUsage(argv[0]);
    if (2 != argc){
        cout<<"Error: parameter error. Exit. "<<endl;
        return -1;
    }
    string dataSetDir = argv[1];
    DataManager dataMgr(dataSetDir);

    SegmentGNet Gnet("Generative Network");
    Gnet.setLearningRate(0.001);
    Gnet.setLossTolerance(0.02);
    Gnet.setBatchSize(3);
    Gnet.initialize();
    Gnet.printArchitecture();

    SegmentDNet Dnet("Discriminative Network");
    Dnet.setLearningRate(0.001);
    Dnet.setLossTolerance(0.02);
    Dnet.setBatchSize(3);
    Dnet.initialize();
    Dnet.printArchitecture();

    StubNetForD stubNet("StubNetwork for Discriminative Network");
    stubNet.setBatchSize(3);
    stubNet.initialize();
    stubNet.printArchitecture();

    Segmentation3DNet gan("3DSegmentationGAN", &Gnet,&Dnet);
    gan.setDataMgr(&dataMgr);
    gan.setStubNet(&stubNet);
    gan.setStubLayer(stubNet.getFinalLayer());

    // pretrain DNet
    int epochsPretrainD = 100;
    for (int i=0; i< epochsPretrainD; ++i){
        gan.pretrainD();
    }

    // train G, D: quick alternative train
    int epochsQuickSwitch = 100;
    for (int i=0; i<epochsQuickSwitch; ++i){
        gan.quicklySwitchTrainG_D();
    }
    // train G, D: slowly alternative train
    int epochsSlowSwitch = 100;
    int epochsAlone = 20;
    for (int i=0; i< epochsSlowSwitch; ++i){
        for(int j=0; j< epochsAlone; ++j){
            gan.trainD();
        }
        for(int j=0; j<epochsAlone; ++j){
            gan.trainG();
        }
    }

    cout<<"==========End of Mnist Test==========="<<endl;
    return 0;
}