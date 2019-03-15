//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.
//

#include "CDLF.h"
#include "VNet.h"

VNet::VNet(const string& saveDir): FeedForwardNet(saveDir){

}
VNet::~VNet(){

}

void VNet::build(){
    // this network uses csv files to load, without direct building.
    return;
}
void VNet::train(){
    InputLayer* inputLayer = getInputLayer();
    Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
    CrossEntropyLoss* lossLayer = (CrossEntropyLoss*) getFinalLayer();

    Tensor<float> groundTruth(lossLayer->m_prevLayer->m_tensorSize);
    groundTruth.uniformInitialize(1);


    int batchSize = getBatchSize();
    int iBatch = 30;

    int i=0;
    while (i< iBatch){
        zeroParaGradient();
        for(int j=0; j<batchSize; ++j){
            generateGaussian(&inputTensor,0,1);
            inputLayer->setInputTensor(inputTensor);
            lossLayer->setGroundTruth(groundTruth);
            forwardPropagate();
            backwardPropagate(true);
            cout<<"Inf: finished one forward and backward propragation at "<<getCurTimeStr()<<endl;
        }
        sgd(getLearningRate(), batchSize);
        ++i;

    }

}
float VNet::test(){
   //null
   return 0;
}
