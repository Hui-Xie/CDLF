
//
// Created by Hui Xie on 01/15/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "DeConvNet.h"


DeConvNet::DeConvNet(const string &name, const string &saveDir) : FeedForwardNet(name, saveDir) {

}

DeConvNet::~DeConvNet() {
    //null
}

void DeConvNet::build() {
    //null: use csv file to create network
}

void DeConvNet::train() {
    InputLayer *inputLayer = getInputLayer();
    SquareLossLayer *lossLayer = (SquareLossLayer *) getFinalLayer();

    vector<int> outputSize = lossLayer->m_prevLayer->m_tensorSize;
    Tensor<float> groundTruthTensor(outputSize);
    for(int i =0; i<groundTruthTensor.getLength(); ++i){
        groundTruthTensor.e(i) = i;
    }
    lossLayer->setGroundTruth(groundTruthTensor);

    const int N = 100;
    const int batchSize = getBatchSize();
    const float learningRate = getLearningRate();
    const int numBatch = (N + batchSize -1) / batchSize;
    int nIter = 0;
    int nBatch = 0;
    while (nBatch < numBatch) {
        zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {
            vector<int> inputSize = inputLayer->m_tensorSize;
            Tensor<float> inputTensor(inputSize);
            generateGaussian(&inputTensor, 0, 1);
            inputLayer->setInputTensor(inputTensor);


            forwardPropagate();
            backwardPropagate(true);
            ++nIter;
        }
        sgd(learningRate, i);
        ++nBatch;
    }
}

float DeConvNet::test() {
    InputLayer *inputLayer = getInputLayer();
    SquareLossLayer *lossLayer = (SquareLossLayer *) getFinalLayer();

    vector<int> outputSize = lossLayer->m_prevLayer->m_tensorSize;
    Tensor<float> groundTruthTensor(outputSize);
    for (int i = 0; i < groundTruthTensor.getLength(); ++i) {
        groundTruthTensor.e(i) = i;
    }
    lossLayer->setGroundTruth(groundTruthTensor);

    float loss = 0;
    const int N = 20;
    int nIter = 0;
    while (nIter < N) {
        vector<int> inputSize = inputLayer->m_tensorSize;
        Tensor<float> inputTensor(inputSize);
        generateGaussian(&inputTensor, 0, 1);
        inputLayer->setInputTensor(inputTensor);

        forwardPropagate();
        loss += lossLayer->getLoss();
        ++nIter;
    }
    printf("output tensor as example:\n");
    lossLayer->m_prevLayer->m_pYTensor->print();
    return loss/N;
}

