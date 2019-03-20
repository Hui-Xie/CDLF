//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.
//

#include "Conv4DNet.h"
#include "CDLF.h"

Conv4DNet::Conv4DNet(const string& saveDir): FeedForwardNet(saveDir){

}
Conv4DNet::~Conv4DNet(){

}

void Conv4DNet::build(){
    //null, build from file
}
void Conv4DNet::train(){
    InputLayer* inputLayer = getInputLayer();
    Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
    LossConvexExample1* lossLayer = (LossConvexExample1*) getFinalLayer();

    int batchSize = getBatchSize();
    int iBatch = 30;

    int i=0;
    while (i< iBatch){
        zeroParaGradient();
        for(int j=0; j<batchSize; ++j){
            generateGaussian(&inputTensor,0,1);
            inputLayer->setInputTensor(inputTensor);
            forwardPropagate();
            backwardPropagate(true);
        }
        averageParaGradient(batchSize);
        optimize("SGD");
        printIteration(lossLayer, i);
        ++i;
    }
    lossLayer->printGroundTruth();
}
float Conv4DNet::test(){
    //null
    return 0;
}
