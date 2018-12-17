//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "NonconvexNet.h"

NonconvexNet::NonconvexNet(const string& name, const string& saveDir, const vector<int>& layerWidthVector): FeedForwardNet(name, saveDir){
  m_layerWidthVector = layerWidthVector;
}

NonconvexNet::~NonconvexNet(){

}

//Notes: this layerWidthVector does not include LossLayer,  and ReLU and NormalizationLayer do not count as a single layer
// Normalization layer generally put after ReLU
void NonconvexNet::build(){
    int nLayers = m_layerWidthVector.size();
    if (0 == nLayers) {
        cout<<"FeedForwardNet has at least one layer."<<endl;
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
    //non-convex example 1: f(x,y) = 3ysin(x)+5xcos(y)+0.5xy+x^2-y^2
    //Notes: Make sure that final layer only 2 neurons.
    //LossNonConvexExample1* lossLayer = new LossNonConvexExample1(10001,"NonConvexLossLayer1", net.getFinalLayer());
    //net.setJudgeLoss(false); //for nonconvex case

    // non-convex example 2: f(x) = x*sin(x)
    //In low -D space, the deep learning network can not escape the the local minima
    // Notes: Make sure that final layer only 1 neuron.
    LossNonConvexExample2* lossLayer = new LossNonConvexExample2(1002,"NonConvexLossLayer2", getFinalLayer());
    addLayer(lossLayer);
}

void NonconvexNet::train(){
    int nIter = 0;
    InputLayer* inputLayer = (InputLayer*)getInputLayer();
    LossLayer* lossLayer = (LossLayer* ) getFinalLayer();
    int maxIteration = 1000;
    int batchSize = getBatchSize();
    float lr = getLearningRate();
    int numBatch =  (maxIteration + batchSize -1 )/ batchSize;

    int nBatch = 0;
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

float NonconvexNet::test(){
   //null
   return 0;
}
