//
// Created by Hui Xie on 12/11/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "MnistAutoEncoder.h"

MnistAutoEncoder::MnistAutoEncoder(const string &name, const string &saveDir) : FeedForwardNet(name, saveDir) {

}

MnistAutoEncoder::~MnistAutoEncoder() {

}

void MnistAutoEncoder::constructGroundTruth(const int labelValue, Tensor<float> &groundTruth) {
    groundTruth.setDimsAndAllocateMem({10, 1});
    groundTruth.zeroInitialize();
    groundTruth.e(labelValue) = 1;
}

void MnistAutoEncoder::build() {
   //null
}


void MnistAutoEncoder::train() {
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();
    float learningRate = getLearningRate();
    zeroParaGradient();
    inputLayer->setInputTensor(m_adversaryTensor);
    lossLayer->setGroundTruth(m_groundTruth);
    forwardPropagate();
    backwardPropagate(false); //do not calculate the gradient of learning parameters of network
    m_adversaryTensor -= (*inputLayer->m_pdYTensor + (m_adversaryTensor- m_originTensor)*m_lambda) * learningRate;  // this method makes adversarial sample try to mimic the desired target.
    //m_adversaryTensor -= inputLayer->m_pdYTensor->sign()* learningRate;  // this methtod makes adversarial sample has many shade
    trimAdversaryTensor();
}


float MnistAutoEncoder::test() {
    //null
    return 0;
}

int MnistAutoEncoder::predict(const Tensor<float>& inputTensor) {
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();
    inputLayer->setInputTensor(inputTensor);
    lossLayer->setGroundTruth(m_groundTruth);
    forwardPropagate();
    int predictValue = lossLayer->m_prevLayer->m_pYTensor->maxPosition();
    return predictValue;
}

void MnistAutoEncoder::trimAdversaryTensor() {
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

void MnistAutoEncoder::saveInputDY(const string filename) {
    InputLayer *inputLayer = getInputLayer();
    inputLayer->m_pdYTensor->save(filename, true);
}

void MnistAutoEncoder::setLambda(float lambda) {
   m_lambda = lambda;
}
