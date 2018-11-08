//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "UnitFilterNet.h"
#include "LossConvexExample1.h"

UnitFilterNet::UnitFilterNet(const string& name, const string& saveDir): FeedForwardNet(name, saveDir){

}

UnitFilterNet::~UnitFilterNet(){

}


void UnitFilterNet::build(){
    int layerID = 1;
    InputLayer* inputLayer = new InputLayer(layerID++, "Input Layer",{3,2,2});
    addLayer(inputLayer);
    ConvolutionLayer* conv1 = new ConvolutionLayer(layerID++, "Conv1", getFinalLayer(), {1,1,1},3);
    addLayer(conv1);
    SubTensorLayer * subTensor = new SubTensorLayer(layerID++, "SubTensor1", getFinalLayer(),{0,0,0,0}, {3,3,2,2});
    addLayer(subTensor);
    LossConvexExample1* loss = new LossConvexExample1(layerID++, "Loss", getFinalLayer());
    addLayer(loss);
}

void UnitFilterNet::train(){
    long nIter = 0;
    InputLayer* inputLayer = (InputLayer*)getInputLayer();
    LossLayer* lossLayer = (LossLayer* ) getFinalLayer();
    long maxIteration = 1000;
    int batchSize = getBatchSize();
    float lr = getLearningRate();
    long numBatch =  maxIteration / batchSize;
    if (0 !=  maxIteration % batchSize){
        numBatch += 1;
    }

    long nBatch = 0;
    while(nBatch < numBatch)
    {
        if (getJudgeLoss() && lossLayer->getLoss()< getLossTolerance()){
            break;
        }
        if (isinf(lossLayer->getLoss())) break;

        zeroParaGradient();
        int i=0;
        for(i=0; i< batchSize && nIter < maxIteration; ++i){
            Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
            generateGaussian(&inputTensor, 0,1);
            inputLayer->setInputTensor(inputTensor);
            forwardPropagate();
            backwardPropagate(true);
            ++nIter;
        }
        sgd(lr,i);
        printIteration(lossLayer, nIter);
        ++nBatch;
    }
    lossLayer->printGroundTruth();
}

float UnitFilterNet::test(){
   //null
   return 0;
}
