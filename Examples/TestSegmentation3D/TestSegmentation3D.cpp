//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "TestSegmentation3D.h"
#include <iostream>
#include "Segmentation3DNet.h"
#include "SegmentGNet.h"
#include "SegmentDNet.h"

using namespace std;

void printUsage(char* argv0){
    cout<<"A Generative Adversarial Network for 3D Medical Images Segmentation: "<<endl;
    cout<<"Usage: "<<endl;

    //todo: add images dirs.
    cout<<argv0<<endl;
}

int main(int argc, char *argv[])
{
    printUsage(argv[0]);
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

    Segmentation3DNet gan("3DSegmentationGAN", &Gnet,&Dnet);
    long epochs = 1000;

    // pretrain D:


    // train G, D
    // quick alternative train
    float accuracy = 0;
    for (long i=0; i<epochs; ++i){
        gan.trainG(1);
        gan.trainD(1);
        accuracy = gan.testG();
        cout<<"Epoch_"<<i<<": "<<" accuracy = "<<accuracy<<endl;
    }
    //slow alternative train





    cout<<"==========End of Mnist Test==========="<<endl;
    return 0;
}