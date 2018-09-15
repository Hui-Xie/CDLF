//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "TestSegmentation3D.h"
#include <iostream>
#include "Segmentation3DNet.h"
using namespace std;

void printUsage(char* argv0){
    cout<<"A Generative Adversarial Network for 3D Medical Image Segmentation: "<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<endl;
}

int main(int argc, char *argv[])
{
    printUsage(argv[0]);
    Segmentation3DNet net;
    net.buildG();
    net.buildD();

    if (! net.checkLayerAttribute()){
        return -1;
    }


    net.setLearningRate(0.001);
    net.setLossTolerance(0.02);
    net.setBatchSize(100);
    net.initialize();



    net.printArchitecture();

    long epoch= 100;
    float accuracy = 0;
    for (long i=0; i<epoch; ++i){
        net.trainG();
        accuracy = net.test();
        cout<<"Epoch_"<<i<<": "<<" accuracy = "<<accuracy<<endl;
    }
    cout<<"==========End of Mnist Test==========="<<endl;
    return 0;


}