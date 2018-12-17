//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "CDLF.h"
#include "PerfNet.h"

PerfNet::PerfNet(const string& name, const string& saveDir): FeedForwardNet(name, saveDir){

}
PerfNet::~PerfNet(){

}

void PerfNet::build(){
    // this network uses csv files to load, without direct building.
    return;
}
void PerfNet::train(){
    InputLayer* inputLayer = getInputLayer();
    Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
    CrossEntropyLoss* lossLayer = (CrossEntropyLoss*) getFinalLayer();

    Tensor<float> groundTruth(lossLayer->m_prevLayer->m_pYTensor->getDims());
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
float PerfNet::test(){
   //null
   return 0;
}
