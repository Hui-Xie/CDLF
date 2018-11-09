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
    cout<<argv0<<"<netDir> <imageAndLabelDir> [outputTestLabelsDir]"<<endl;
    cout<<"Where"<<endl;
    cout<<"netDir: the net parameters saved diretory"<<endl;
    cout<<"the imageAndLabelDir must include 4 subdirectories: testImages  testLabels  trainImages  trainLabels" <<endl;
    cout<<"And the corresponding images file and label file should have same filename in different directories. "<<endl;
    cout<<"outputTestLabelsDir is the directory for outputting test label files"<<endl;
    cout<<"Input parameter example: /Users/hxie1/msd/Task07_Pancreas/CDLFData /Users/hxie1/temp_3DGANOuput"<<endl;
}

int main(int argc, char *argv[]) {
    //cout<<"3D segmentation for One Sample";
    printCurrentLocalTime();
    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout << "Info: program use CPU, instead of GPU." << endl;
#endif


    printUsage(argv[0]);
    if (3 != argc && 4 != argc) {
        cout << "Error: parameter error. Exit. " << endl;
        return -1;
    }
    string netDir = argv[1];
    string dataSetDir = argv[2];
    string outputLabelsDir = "";
    if (4 == argc) {
        outputLabelsDir = argv[3];
    }
    DataManager dataMgr(dataSetDir, outputLabelsDir);

    SegmentGNet Gnet("GNet", netDir);
    if (isEmptyDir(Gnet.getDir())) {
        Gnet.build();
        Gnet.initialize();
        Gnet.setLearningRate(0.001);
        Gnet.setLossTolerance(0.02);
        Gnet.setBatchSize(3);

    } else {
        Gnet.load();
    }
    Gnet.printArchitecture();

    SegmentDNet Dnet("DNet", netDir);
    if (isEmptyDir(Dnet.getDir())) {
        Dnet.build();
        Dnet.initialize();
        Dnet.setLearningRate(0.001);
        Dnet.setLossTolerance(0.02);
        Dnet.setBatchSize(3);
    }
    else{
        Dnet.load();
    }
    Dnet.printArchitecture();

    StubNetForD stubNet("StubNet", netDir);
    if (isEmptyDir(stubNet.getDir())) {
        stubNet.build();
        stubNet.initialize();
        stubNet.setBatchSize(3);
    }
    else{
        stubNet.load();
    }
    stubNet.printArchitecture();

    Gnet.save(); Dnet.save(); stubNet.save();

    Segmentation3DNet gan("3DSegmentationGAN", &Gnet,&Dnet);
    gan.setDataMgr(&dataMgr);
    gan.setStubNet(&stubNet);
    gan.setStubLayer(stubNet.getFinalLayer());

    // pretrain DNet
    cout<<"Info: start pretrain D "<<endl;
    printCurrentLocalTime();
    int epochsPretrainD = 5; //100;
    for (int i=0; i< epochsPretrainD; ++i){
        gan.pretrainD();
        printf("Pre-train D at %d of %d, ", i, epochsPretrainD);
        printCurrentLocalTime();
    }
    Gnet.save(); Dnet.save(); stubNet.save();


    // train G, D: quick alternative train
    cout<<"Info: start quick switch to train G and D "<<endl;
    printCurrentLocalTime();
    int epochsQuickSwitch = 10; //100;
    for (int i=0; i<epochsQuickSwitch; ++i){
        gan.quicklySwitchTrainG_D();
        printf("Quick switch train G_D at %d of %d, ", i,  epochsQuickSwitch);
        printCurrentLocalTime();
    }
    Gnet.save(); Dnet.save(); stubNet.save();

    // train G, D: slowly alternative train
    cout<<"Info: start slow switch to train G and D "<<endl;
    printCurrentLocalTime();
    int epochsSlowSwitch = 10;//100;
    int epochsAlone = 5;// 20;
    for (int i=0; i< epochsSlowSwitch; ++i){
        for(int j=0; j< epochsAlone; ++j){
            gan.trainD();
            printf("Slow  switch train D at %d of %d, in %d of %d, ", j,  epochsAlone, i, epochsSlowSwitch);
            printCurrentLocalTime();
        }
        Dnet.save();
        for(int j=0; j<epochsAlone; ++j){
            gan.trainG();
            printf("Slow  switch train G at %d of %d, in %d of %d, ", j,  epochsAlone, i, epochsSlowSwitch);
            printCurrentLocalTime();
        }
        Gnet.save();

        cout<<"Slow Switch Epoch: "<<i<<endl;
        printCurrentLocalTime();
        if (i != epochsSlowSwitch -1){
            gan.testG(false);
        }
        else{
            gan.testG(true); //output file
        }

        printf("Test G at %d of %d, ", i,  epochsSlowSwitch);
        printCurrentLocalTime();

    }
    Gnet.save(); Dnet.save(); stubNet.save();
    cout<< "=========== End of Test:  "<<gan.getName() <<" ============"<<endl;
    printCurrentLocalTime();
    return 0;
}