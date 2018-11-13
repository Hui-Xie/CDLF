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
    inputLayer->setInputTensor(m_adversaryTensor);
    lossLayer->setGroundTruth(m_groundTruth);
    forwardPropagate();
    backwardPropagate(false); //do not calculate the gradient of learning parameters of network
    m_adversaryTensor -= (*inputLayer->m_pdYTensor + (m_adversaryTensor- m_originTensor)*m_lambda) * learningRate;
    trimAdversaryTensor();
}


float AdverMnistNet::test() {
    //null
    return 0;
}

int AdverMnistNet::predict(const Tensor<float>& inputTensor) {
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();
    inputLayer->setInputTensor(inputTensor);
    lossLayer->setGroundTruth(m_groundTruth);
    forwardPropagate();
    int predictValue = lossLayer->m_prevLayer->m_pYTensor->maxPosition();
    return predictValue;
}

void AdverMnistNet::trimAdversaryTensor() {
    int N = m_adversaryTensor.getLength();
    for(int i=0; i<N; ++i){
        if (m_adversaryTensor.e(i) < 0){
            m_adversaryTensor.e(i) =0;
        }
        if (m_adversaryTensor.e(i) >255){
            m_adversaryTensor.e(i) =255;
        }
    }

}

void AdverMnistNet::saveInputDY(const string filename) {
    InputLayer *inputLayer = getInputLayer();
    inputLayer->m_pdYTensor->save(filename, true);
}

void AdverMnistNet::setLambda(float lambda) {
   m_lambda = lambda;
}
