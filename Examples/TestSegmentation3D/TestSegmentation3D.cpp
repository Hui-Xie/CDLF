//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "TestSegmentation3D.h"
#include <iostream>
#include "Segmentation3DNet.h"
using namespace std;

void printUsage(char* argv0){
    cout<<"Test 3D Medical Image Segmentation: "<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" fullPathOfImagesDir"<<endl;
}

int main(int argc, char *argv[]) {

    if (2 != argc) {
        cout << "Error: input parameter error." << endl;
        printUsage(argv[0]);
        return -1;
    }
    Segmentation3DNet net;

    net.build();
    net.setLearningRate(0.001);
    net.setLossTolerance(0.02);
    net.setBatchSize(100);
    net.initialize();

    net.printArchitecture();

    long epoch= 100;
    float accuracy = 0;
    for (long i=0; i<epoch; ++i){
        net.train();
        accuracy = net.test();
        cout<<"Epoch_"<<i<<": "<<" accuracy = "<<accuracy<<endl;
    }
    cout<<"==========End of Mnist Test==========="<<endl;
    return 0;


}