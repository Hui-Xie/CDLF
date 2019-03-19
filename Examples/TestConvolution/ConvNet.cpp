//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.
//

#include "ConvNet.h"


ConvNet::ConvNet(const string& saveDir): FeedForwardNet(saveDir){

}
ConvNet::~ConvNet(){

}

void ConvNet::build(){
   //null, build from csv file
}
void ConvNet::train(){
    InputLayer* inputLayer = getInputLayer();
    Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
    SquareLossLayer* lossLayer = (SquareLossLayer*) getFinalLayer();
    Tensor<float> groundTruth(lossLayer->m_prevLayer->m_tensorSize);
    for (int i=0; i< groundTruth.getLength(); ++i){
        groundTruth.e(i) = i;
    }
    lossLayer->setGroundTruth(groundTruth);

    int batchSize = getBatchSize();
    int iBatch = 130;

    int i=0;
    while (i< iBatch){
        zeroParaGradient();
        for(int j=0; j<batchSize; ++j){
            generateGaussian(&inputTensor,0,1);
            inputLayer->setInputTensor(inputTensor);

            forwardPropagate();
            backwardPropagate(true);
        }
        sgd(getLearningRate());
        printIteration(lossLayer, i);
        ++i;
    }
    lossLayer->printGroundTruth();
}
float ConvNet::test(){
   //null
   return 0;
}
