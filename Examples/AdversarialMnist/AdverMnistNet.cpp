//
// Created by Hui Xie on 11/9/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "AdverMnistNet.h"
#include "CDLF.h"

AdverMnistNet::AdverMnistNet(const string &name, const string &saveDir) : FeedForwardNet(name, saveDir) {

}

AdverMnistNet::~AdverMnistNet() {

}

void AdverMnistNet::constructGroundTruth(const int labelValue, Tensor<float> &groundTruth) {
    groundTruth.setDimsAndAllocateMem({10, 1});
    groundTruth.zeroInitialize();
    groundTruth.e(labelValue) = 1;
}

void AdverMnistNet::build() {
   //null
}


void AdverMnistNet::train() {
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();
    float learningRate = getLearningRate();
    zeroParaGradient();
    inputLayer->setInputTensor(m_inputTensor);
    lossLayer->setGroundTruth(m_groundTruth);
    forwardPropagate();
    backwardPropagate(false); //do not calculate the gradient of learning parameters
    m_inputTensor -= (*inputLayer->m_pdYTensor) * learningRate;
}


float AdverMnistNet::test() {
    //null
    return 0;
}

int AdverMnistNet::predict() {
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();
    inputLayer->setInputTensor(m_inputTensor);
    lossLayer->setGroundTruth(m_groundTruth);
    forwardPropagate();
    int predictValue = lossLayer->m_prevLayer->m_pYTensor->maxPosition();
    return predictValue;
}
