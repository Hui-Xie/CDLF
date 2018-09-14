//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "NonconvexNet.h"

NonconvexNet::NonconvexNet(const vector<long>& layerWidthVector){
  m_layerWidthVector = layerWidthVector;
}

NonconvexNet::~NonconvexNet(){

}

//Notes: this layerWidthVector does not include LossLayer,  and ReLU and NormalizationLayer do not count as a single layer
// Normalization layer generally put after ReLU
void NonconvexNet::build(){
    int nLayers = m_layerWidthVector.size();
    if (0 == nLayers) {
        cout<<"Net has at least one layer."<<endl;
        return;
    }
    int layerID = 1;
    InputLayer* inputLayer = new InputLayer(layerID++, "Input Layer",{m_layerWidthVector.at(0),1});
    addLayer(inputLayer);
    for(int i =1; i< nLayers; ++i){
        FCLayer* fcLayer = new FCLayer(layerID++, "FCLayer"+to_string(i), getFinalLayer(), m_layerWidthVector.at(i));
        addLayer(fcLayer);
        if (i != nLayers -1){
            ReLU* reLU = new ReLU(layerID++, "ReLU"+to_string(i), getFinalLayer());
            addLayer(reLU);
            NormalizationLayer* normalLayer = new NormalizationLayer(layerID++, "NormLayer"+to_string(i),getFinalLayer());
            addLayer(normalLayer);
        }
    }
}

void NonconvexNet::train(){
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
            backwardPropagate();
            ++nIter;
        }
        sgd(lr,i);
        printIteration(lossLayer, nIter);
        ++nBatch;
    }
    lossLayer->printGroundTruth();
}

float NonconvexNet::test(){
   //null
   return 0;
}
